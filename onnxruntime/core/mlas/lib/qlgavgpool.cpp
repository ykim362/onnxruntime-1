/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    quantize.cpp

Abstract:

    This module implements routines to quantize buffers.

    For quantization formula as specified in the ONNX operator documentation is:

        Output = Saturate(RoundToEven(Input / Scale) + ZeroPoint)

--*/

#include "quantize.h"
#include <cassert>

size_t
MLASCALL
MlasQLinearSafePaddingElementCount(
    size_t ElementSize,
    size_t ElementCount
    )
{
    assert(ElementSize == 1 || ElementSize == 2 || ElementSize == 4 || ElementSize == 8 || ElementSize == 16);
    return ElementCount + (size_t{256} / ElementSize - 1);
}

#if defined(MLAS_NEON_INTRINSICS)

static
bool
MlasCalculateParametersForGloabalAveragePool(
    size_t ImageSize,
    float ScaleInput,
    int32_t ZeroPointInput,
    float ScaleOutput,
    int32_t& Bias,
    int32_t& Multiplier,
    int32_t& Shift,
    uint64_t& Rounding
    )
{
    Bias = - ZeroPointInput * gsl::narrow_cast<int32_t>(ImageSize);
    float scale = ScaleInput / (ScaleOutput * (float)ImageSize);
    if (scale < 0x1.0p-32f || scale >=  256.0f) return false;

    const uint32_t scale_bits = MlasBitsOfFp32(scale);
    Multiplier = (int32_t)(scale_bits & 0x007FFFFF | 0x00800000);
    if (Multiplier < 0x00800000 || Multiplier > 0x00FFFFFF) return false;

    // Shift is in [16, 55] range.
    Shift = 127 + 23 - (scale_bits >> 23);
    if (Shift < 16 || Shift > 55) return false;
    Rounding = uint64_t{1} << ((uint32_t)Shift - 1);

    return true;
}

#elif defined(MLAS_SSE2_INTRINSICS)

void
MLASCALL
MlasQLinearGlobalAveragePool(
    const uint8_t* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    uint8_t* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Channels,
    size_t ImageSize
    )
{
    const auto vscale = MlasBroadcastFloat32x4(ScaleInput / static_cast<float>(ImageSize) / ScaleOutput);
    const auto vbias = MlasBroadcastInt32x4(-ZeroPointInput * gsl::narrow_cast<int32_t>(ImageSize));
    const auto vmin_value = MlasBroadcastFloat32x4(float(0 - ZeroPointOutput));
    const auto vmax_value = MlasBroadcastFloat32x4(float(255 - ZeroPointOutput));
    const auto vzero_point = MlasBroadcastInt32x4(ZeroPointOutput);
    const auto vzero = _mm_setzero_si128();
    uint8_t buffer[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    int32_t sum_buffer[8];
    int sum_count = 0;
    for (; Channels > 0; Channels--) {
        __m128i vacc_lo = _mm_setzero_si128();
        __m128i vacc_hi = _mm_setzero_si128();
        auto Len = ImageSize;
        for (; Len >= 32; Len -= 32) {
            const __m128i vi0 = _mm_loadl_epi64((const __m128i*)Input);
            const __m128i vi1 = _mm_loadl_epi64((const __m128i*)(Input + 8));
            const __m128i vi2 = _mm_loadl_epi64((const __m128i*)(Input + 16));
            const __m128i vi3 = _mm_loadl_epi64((const __m128i*)(Input + 24));

            const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
            const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
            const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
            const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);

            const __m128i vsum = _mm_add_epi16(_mm_add_epi16(vxi0, vxi1), _mm_add_epi16(vxi2, vxi3));
            vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vsum, vzero));
            vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vsum, vzero));
            Input += 32;
        }
        for (; Len >= 8; Len -= 8) {
            const __m128i vsum = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)Input), vzero);
            vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vsum, vzero));
            vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vsum, vzero));
            Input += 8;
        }
        if (Len > 0) {
            _mm_storel_epi64((__m128i*)buffer, vzero);
            MlasCopyTailBytes(buffer, Input, Len);
            const __m128i vsum = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)buffer), vzero);
            vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vsum, vzero));
            vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vsum, vzero));
            Input += Len;
        }

        __m128i vacc = _mm_add_epi32(vacc_lo, vacc_hi);                    // [ D C | B A ]
        __m128i vshuf = _mm_shuffle_epi32(vacc, _MM_SHUFFLE(2, 3, 0, 1));  // [ C D | A B ]
        __m128i vsums = _mm_add_epi32(vacc, vshuf);                        // [ D+C C+D | B+A A+B ]
        vshuf = _mm_shuffle_epi32(vsums, _MM_SHUFFLE(1, 0, 3, 2));         // [ B+A A+B | D+C C+D ]
        vsums = _mm_add_epi32(vsums, vshuf);
        sum_buffer[sum_count++] = _mm_cvtsi128_si32(vsums);

        if (sum_count == 8) {
            auto vresult0 = _mm_load_si128((const __m128i*)sum_buffer);
            auto vresult1 = _mm_load_si128(((const __m128i*)sum_buffer) + 1);
            vresult0 = MlasRequantizeOutputVector(vresult0, vbias, vscale, vmin_value, vmax_value, vzero_point);
            vresult1 = MlasRequantizeOutputVector(vresult1, vbias, vscale, vmin_value, vmax_value, vzero_point);
            vresult0 = _mm_packus_epi16(vresult0, vresult1);
            vresult0 = _mm_packus_epi16(vresult0, vresult0);
            _mm_storel_epi64((__m128i*)Output, vresult0);
            Output += 8;
            sum_count = 0;
        }
    }
    if (sum_count > 0) {
        auto vresult0 = _mm_load_si128((const __m128i*)sum_buffer);
        auto vresult1 = _mm_load_si128(((const __m128i*)sum_buffer) + 1);
        vresult0 = MlasRequantizeOutputVector(vresult0, vbias, vscale, vmin_value, vmax_value, vzero_point);
        vresult1 = MlasRequantizeOutputVector(vresult1, vbias, vscale, vmin_value, vmax_value, vzero_point);
        vresult0 = _mm_packus_epi16(vresult0, vresult1);
        vresult0 = _mm_packus_epi16(vresult0, vresult0);
        if (sum_count >= 4) {
            *(int32_t*)Output = _mm_cvtsi128_si32(vresult0);
            Output += 4;
            sum_count -= 4;
            vresult0 = _mm_shuffle_epi32(vresult0, _MM_SHUFFLE(0, 3, 2, 1));
        }
        if (sum_count > 0) {
            uint32_t tail_values = (uint32_t)_mm_cvtsi128_si32(vresult0);
            for (; sum_count > 0; --sum_count) {
                *Output++ = (uint8_t)tail_values;
                tail_values = tail_values >> 8;
            }
        }
    }
}

static
void
MLASCALL
MlasNhwcQLinearGlobalAveragePoolSingleBatch(
    const uint8_t* Input,
    uint8_t* Output,
    const uint8_t* InputEnd,
    size_t ImageSize,
    size_t Channels,
    size_t Stride,
    const __m128i vbias,
    const __m128 vscale,
    const __m128 vmin_value,
    const __m128 vmax_value,
    const __m128i vzero_point,
    int32_t* AccumulateBuffer,
    const uint8_t* ZeroBuffer
    )
{
    uint8_t tail_buffer[8];
    bool finish_one_pass = false;
    const __m128i vzero = _mm_setzero_si128();

#define LOAD_FULL_CHANNELS() \
    const __m128i vi0 = _mm_loadl_epi64((const __m128i*)i0); \
    i0 += 8; \
    const __m128i vi1 = _mm_loadl_epi64((const __m128i*)i1);\
    i1 += 8;\
    const __m128i vi2 = _mm_loadl_epi64((const __m128i*)i2);\
    i2 += 8;\
    const __m128i vi3 = _mm_loadl_epi64((const __m128i*)i3);\
    i3 += 8;\
    const __m128i vi4 = _mm_loadl_epi64((const __m128i*)i4);\
    i4 += 8;\
    const __m128i vi5 = _mm_loadl_epi64((const __m128i*)i5);\
    i5 += 8;\
    const __m128i vi6 = _mm_loadl_epi64((const __m128i*)i6);\
    i6 += 8;

#define CACULATE_ACCUMULATE_VECTORS()                                                                  \
    __m128i vacc_lo = finish_one_pass ? _mm_load_si128((__m128i*)acc) : vbias;                         \
    __m128i vacc_hi = finish_one_pass ? _mm_load_si128((__m128i*)acc + 1) : vbias;                     \
    const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);                                                \
    const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);                                                \
    const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);                                                \
    const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);                                                \
    const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);                                                \
    const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);                                                \
    const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);                                                \
    const __m128i vsum01 = _mm_add_epi16(vxi0, vxi1);                                                  \
    const __m128i vsum23 = _mm_add_epi16(vxi2, vxi3);                                                  \
    const __m128i vsum45 = _mm_add_epi16(vxi4, vxi5);                                                  \
    const __m128i vsum016 = _mm_add_epi16(vsum01, vxi6);                                               \
    const __m128i vsum2345 = _mm_add_epi16(vsum23, vsum45);                                            \
    const __m128i vsum = _mm_add_epi16(vsum016, vsum2345);                                             \
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vsum, vzero));                                 \
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vsum, vzero))

    const uint8_t* group_start = Input;
    for (; ImageSize > 7; ImageSize -= 7) {
        const uint8_t* i0 = group_start;
        const uint8_t* i1 = group_start + Stride;
        const uint8_t* i2 = group_start + Stride * 2;
        const uint8_t* i3 = group_start + Stride * 3;
        const uint8_t* i4 = group_start + Stride * 4;
        const uint8_t* i5 = group_start + Stride * 5;
        const uint8_t* i6 = group_start + Stride * 6;

        size_t channels_remaining = Channels;
        int32_t* acc = AccumulateBuffer;
        for (; channels_remaining >= 8; channels_remaining -= 8) {
            LOAD_FULL_CHANNELS();
            CACULATE_ACCUMULATE_VECTORS();
            _mm_store_si128((__m128i*)acc, vacc_lo);
            _mm_store_si128((__m128i*)acc + 1, vacc_hi);
            acc += 8;
        }

        if (channels_remaining > 0) {
            uint8_t* tb = &tail_buffer[0];
            const uint8_t* LastInputOf8 = InputEnd - 8;
            auto SafeTailBuffer = [tb, channels_remaining, LastInputOf8](const uint8_t* p) {
                if (p >= LastInputOf8) {
                    std::copy_n(p, channels_remaining, tb);
                    return (const uint8_t*)tb;
                }
                return p;
            };

            const __m128i vi0 = _mm_loadl_epi64((const __m128i*)SafeTailBuffer(i0));
            const __m128i vi1 = _mm_loadl_epi64((const __m128i*)SafeTailBuffer(i1));
            const __m128i vi2 = _mm_loadl_epi64((const __m128i*)SafeTailBuffer(i2));
            const __m128i vi3 = _mm_loadl_epi64((const __m128i*)SafeTailBuffer(i3));
            const __m128i vi4 = _mm_loadl_epi64((const __m128i*)SafeTailBuffer(i4));
            const __m128i vi5 = _mm_loadl_epi64((const __m128i*)SafeTailBuffer(i5));
            const __m128i vi6 = _mm_loadl_epi64((const __m128i*)SafeTailBuffer(i6));

            CACULATE_ACCUMULATE_VECTORS();

            _mm_store_si128((__m128i*)acc, vacc_lo);
            _mm_store_si128((__m128i*)acc + 1, vacc_hi);
        }

        finish_one_pass = true;
        group_start += 7 * Stride;
    }

    if (ImageSize > 0) {
        const uint8_t* i0 = group_start;
        const uint8_t* i1 = (ImageSize > 1 ? group_start + Stride : ZeroBuffer);
        const uint8_t* i2 = (ImageSize > 2 ? group_start + Stride * 2 : ZeroBuffer);
        const uint8_t* i3 = (ImageSize > 3 ? group_start + Stride * 3 : ZeroBuffer);
        const uint8_t* i4 = (ImageSize > 4 ? group_start + Stride * 4 : ZeroBuffer);
        const uint8_t* i5 = (ImageSize > 5 ? group_start + Stride * 5 : ZeroBuffer);
        const uint8_t* i6 = (ImageSize > 6 ? group_start + Stride * 6 : ZeroBuffer);

        int32_t* acc = AccumulateBuffer;
        for (; Channels >= 8; Channels -= 8) {
            LOAD_FULL_CHANNELS();
            CACULATE_ACCUMULATE_VECTORS();
            acc += 8;

            vacc_lo = MlasRequantizeOutputVector(vacc_lo, vzero, vscale, vmin_value, vmax_value, vzero_point);
            vacc_hi = MlasRequantizeOutputVector(vacc_hi, vzero, vscale, vmin_value, vmax_value, vzero_point);
            __m128i vresult = _mm_packus_epi16(vacc_lo, vacc_hi);
            vresult = _mm_packus_epi16(vresult, vresult);
            _mm_storel_epi64((__m128i*)Output, vresult);
            Output += 8;
        }

        if (Channels > 0) {
            uint8_t* tb = &tail_buffer[0];
            const uint8_t* LastInputOf8 = InputEnd - 8;
            auto SafeTailBuffer = [tb, Channels, LastInputOf8, ImageSize](const uint8_t* p, size_t idx) {
                if (idx < ImageSize && p >= LastInputOf8) {
                    std::copy_n(p, Channels, tb);
                    return (const uint8_t*)tb;
                }
                return p;
            };

            const __m128i vi0 = _mm_loadl_epi64((const __m128i*)SafeTailBuffer(i0, 0));
            const __m128i vi1 = _mm_loadl_epi64((const __m128i*)SafeTailBuffer(i1, 1));
            const __m128i vi2 = _mm_loadl_epi64((const __m128i*)SafeTailBuffer(i2, 2));
            const __m128i vi3 = _mm_loadl_epi64((const __m128i*)SafeTailBuffer(i3, 3));
            const __m128i vi4 = _mm_loadl_epi64((const __m128i*)SafeTailBuffer(i4, 4));
            const __m128i vi5 = _mm_loadl_epi64((const __m128i*)SafeTailBuffer(i5, 5));
            const __m128i vi6 = _mm_loadl_epi64((const __m128i*)SafeTailBuffer(i6, 6));

            CACULATE_ACCUMULATE_VECTORS();

            vacc_lo = MlasRequantizeOutputVector(vacc_lo, vzero, vscale, vmin_value, vmax_value, vzero_point);
            vacc_hi = MlasRequantizeOutputVector(vacc_hi, vzero, vscale, vmin_value, vmax_value, vzero_point);
            __m128i vresult = _mm_packus_epi16(vacc_lo, vacc_hi);
            vresult = _mm_packus_epi16(vresult, vresult);

            if (Channels >= 4) {
                *(int32_t*)Output = _mm_cvtsi128_si32(vresult);
                vresult = _mm_shuffle_epi32(vresult, _MM_SHUFFLE(0, 3, 2, 1));
                Output += 4;
                Channels -= 4;
            }
            unsigned int b4 = _mm_cvtsi128_si32(vresult);
            for (; Channels > 0; Channels--) {
                *Output++ = static_cast<uint8_t>(b4);
                b4 = b4 >> 8;
            }
        }
    }
}

void
MLASCALL
MlasNhwcQLinearGlobalAveragePool(
    const uint8_t* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    uint8_t* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Batch,
    size_t ImageSize,
    size_t Stride,
    size_t Channels,
    int32_t* AccumulateBuffer,
    const uint8_t* ZeroBuffer
    )
{
    if (Stride == 1) {
        assert(Channels == Stride);
        MlasQLinearGlobalAveragePool(Input, ScaleInput, ZeroPointInput, Output, ScaleOutput, ZeroPointOutput, Batch, ImageSize);
        return;
    }

    const uint8_t* InputEnd = Input + (Batch * ImageSize * Stride - Stride + Channels);
    int32_t Bias = -ZeroPointInput * gsl::narrow_cast<int32_t>(ImageSize);
    const auto vbias = MlasBroadcastInt32x4(Bias);
    float scale = ScaleInput / gsl::narrow_cast<float>(ImageSize) / ScaleOutput;
    const auto vscale = MlasBroadcastFloat32x4(scale);
    const auto vmin_value = MlasBroadcastFloat32x4(float(0 - ZeroPointOutput));
    const auto vmax_value = MlasBroadcastFloat32x4(float(255 - ZeroPointOutput));
    const auto vzero_point = MlasBroadcastInt32x4(ZeroPointOutput);

    for (; Batch > 0; Batch--) {
        MlasNhwcQLinearGlobalAveragePoolSingleBatch(
            Input, Output, InputEnd, ImageSize, Channels, Stride,
            vbias, vscale, vmin_value, vmax_value, vzero_point,
            AccumulateBuffer, ZeroBuffer);
        Input += ImageSize * Stride;
        Output += Stride;
    }
}

#elif

// Pure C++ Implementation

void
MLASCALL
MlasQLinearGlobalAveragePool(
    const uint8_t* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    uint8_t* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Channels,
    size_t ImageSize
    )
{
    int32_t bias = -ZeroPointInput * gsl::narrow_cast<int32_t>(ImageSize);
    float scale = ScaleInput / (scaleOutput * static_cast<float>(ImageSize));
    for (; Channels > 0; Channels--) {
        int32_t acc = bias;
        for (size_t i = 0; i < ImageSize; ++i) {
            acc += static_cast<int>(*Input++);
        }
        int32_t v = static_cast<int>(std::nearbyinf(acc * scale)) + ZeroPointOutput;
        *Output++ = std::max(std::min(255, v), 0);
    }
}

void
MLASCALL
MlasNhwcQLinearGlobalAveragePool(
    const uint8_t* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    uint8_t* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Batch,
    size_t ImageSize,
    size_t Stride,
    size_t Channels,
    int32_t* AccumulateBuffer,
    const uint8_t* ZeroBuffer
    )
{
    int32_t bias = -ZeroPointInput * gsl::narrow_cast<int32_t>(ImageSize);
    float scale = ScaleInput / (scaleOutput * static_cast<float>(ImageSize));
    for (; Batch > 0; Batch--) {
        const uint8_t* batch_input = Input;
        uint8_t* batch_output = Output;
        Input += Stride * ImageSize;
        Output += Stride;
        std::fill_n(AccumulateBuffer, Channels, bias);
        for (size_t i = 0; i < ImageSize; ++i) {
            for (size_t c = 0; c < Channels; ++c) {
              AccumulateBuffer[c] += static_cast<int>(batch_input[c]);
            }
            batch_input += Stride;
        }
        for (size_t c = 0; c < Channels; ++c) {
            int32_t v = static_cast<int>(std::nearbyinf(AccumulateBuffer[c] * scale)) + ZeroPointOutput;
            *batch_output++ = std::max(std::min(255, v), 0);
        }
    }
}

#endif
