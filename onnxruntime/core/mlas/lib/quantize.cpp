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

#include "mlasi.h"
#include <cassert>

#if defined(MLAS_NEON64_INTRINSICS) || defined(MLAS_SSE2_INTRINSICS)

//
// QuantizeLinear implementation using NEON or SSE2 intrinsics.
//

MLAS_FORCEINLINE
MLAS_INT32X4
MlasQuantizeLinearVector(
    MLAS_FLOAT32X4 FloatVector,
    MLAS_FLOAT32X4 ScaleVector,
    MLAS_FLOAT32X4 MinimumValueVector,
    MLAS_FLOAT32X4 MaximumValueVector,
    MLAS_INT32X4 ZeroPointVector
    )
{
    //
    // Scale the input vector and clamp the values to the minimum and maximum
    // range (adjusted by the zero point value).
    //

    FloatVector = MlasDivideFloat32x4(FloatVector, ScaleVector);

#if defined(MLAS_NEON64_INTRINSICS)
    // N.B. FMINNM and FMAXNM returns the numeric value if either of the values
    // is a NaN.
    FloatVector = vmaxnmq_f32(FloatVector, MinimumValueVector);
    FloatVector = vminnmq_f32(FloatVector, MaximumValueVector);
#else
    // N.B. MINPS and MAXPS returns the value from the second vector if the
    // value from the first vector is a NaN.
    FloatVector = _mm_max_ps(FloatVector, MinimumValueVector);
    FloatVector = _mm_min_ps(FloatVector, MaximumValueVector);
#endif

    //
    // Convert the float values to integer using "round to nearest even" and
    // then shift the output range using the zero point value.
    //

#if defined(MLAS_NEON64_INTRINSICS)
    auto IntegerVector = vcvtnq_s32_f32(FloatVector);
    IntegerVector = vaddq_s32(IntegerVector, ZeroPointVector);
#else
    // N.B. Assumes MXCSR has been configured with the default rounding mode of
    // "round to nearest even".
    auto IntegerVector = _mm_cvtps_epi32(FloatVector);
    IntegerVector = _mm_add_epi32(IntegerVector, ZeroPointVector);
#endif

    return IntegerVector;
}

template<typename OutputType>
MLAS_INT32X4
MlasQuantizeLinearPackBytes(
    MLAS_INT32X4 IntegerVector
    );

#if defined(MLAS_NEON64_INTRINSICS)

template<typename OutputType>
MLAS_INT32X4
MlasQuantizeLinearPackBytes(
    MLAS_INT32X4 IntegerVector
    )
{
    //
    // Swizzle the least significant byte from each int32_t element to the
    // bottom four bytes of the vector register.
    //

    uint16x8_t WordVector = vreinterpretq_u16_s32(IntegerVector);
    WordVector = vuzp1q_u16(WordVector, WordVector);
    uint8x16_t ByteVector = vreinterpretq_u8_u16(WordVector);
    ByteVector = vuzp1q_u8(ByteVector, ByteVector);

    return vreinterpretq_s32_u8(ByteVector);
}

#else

template<>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasQuantizeLinearPackBytes<uint8_t>(
    MLAS_INT32X4 IntegerVector
    )
{
    IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);
    IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);

    return IntegerVector;
}

template<>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasQuantizeLinearPackBytes<int8_t>(
    MLAS_INT32X4 IntegerVector
    )
{
    IntegerVector = _mm_packs_epi16(IntegerVector, IntegerVector);
    IntegerVector = _mm_packs_epi16(IntegerVector, IntegerVector);

    return IntegerVector;
}

#endif

template<typename OutputType>
void
MLASCALL
MlasQuantizeLinear(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint
    )
/*++

Routine Description:

    This routine quantizes the input buffer using the supplied quantization
    parameters.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    constexpr int32_t MinimumValue = std::numeric_limits<OutputType>::min();
    constexpr int32_t MaximumValue = std::numeric_limits<OutputType>::max();

    auto ScaleVector = MlasBroadcastFloat32x4(Scale);
    auto MinimumValueVector = MlasBroadcastFloat32x4(float(MinimumValue - ZeroPoint));
    auto MaximumValueVector = MlasBroadcastFloat32x4(float(MaximumValue - ZeroPoint));
    auto ZeroPointVector = MlasBroadcastInt32x4(ZeroPoint);

    while (N >= 4) {

        auto FloatVector = MlasLoadFloat32x4(Input);
        auto IntegerVector = MlasQuantizeLinearVector(FloatVector, ScaleVector,
            MinimumValueVector, MaximumValueVector, ZeroPointVector);

        IntegerVector = MlasQuantizeLinearPackBytes<OutputType>(IntegerVector);

#if defined(MLAS_NEON64_INTRINSICS)
        vst1q_lane_s32((int32_t*)Output, IntegerVector, 0);
#else
        *((int32_t*)Output) = _mm_cvtsi128_si32(IntegerVector);
#endif

        Input += 4;
        Output += 4;
        N -= 4;
    }

    for (size_t n = 0; n < N; n++) {

#if defined(MLAS_NEON64_INTRINSICS)
        auto FloatVector = vld1q_dup_f32(Input + n);
#else
        auto FloatVector = _mm_load_ss(Input + n);
#endif
        auto IntegerVector = MlasQuantizeLinearVector(FloatVector, ScaleVector,
            MinimumValueVector, MaximumValueVector, ZeroPointVector);

#if defined(MLAS_NEON64_INTRINSICS)
        vst1q_lane_u8((uint8_t*)Output + n, vreinterpretq_u8_s32(IntegerVector), 0);
#else
        *((uint8_t*)Output + n) = (uint8_t)_mm_cvtsi128_si32(IntegerVector);
#endif
    }
}

#else

//
// QuantizeLinear implementation using the C++ runtime.
//

template<typename OutputType>
void
MLASCALL
MlasQuantizeLinear(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint
    )
/*++

Routine Description:

    This routine quantizes the input buffer using the supplied quantization
    parameters.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    constexpr int32_t MinimumValue = std::numeric_limits<OutputType>::min();
    constexpr int32_t MaximumValue = std::numeric_limits<OutputType>::max();

    for (size_t n = 0; n < N; n++) {

        float FloatValue = std::nearbyintf(Input[n] / Scale) + float(ZeroPoint);
        FloatValue = std::max(FloatValue, float(MinimumValue));
        FloatValue = std::min(FloatValue, float(MaximumValue));
        Output[n] = (OutputType)(int32_t)FloatValue;
    }
}

#endif

template
void
MLASCALL
MlasQuantizeLinear<int8_t>(
    const float* Input,
    int8_t* Output,
    size_t N,
    float Scale,
    int8_t ZeroPoint
    );

template
void
MLASCALL
MlasQuantizeLinear<uint8_t>(
    const float* Input,
    uint8_t* Output,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    );

#if defined(MLAS_SSE2_INTRINSICS)

MLAS_FORCEINLINE
MLAS_INT32X4
MlasRequantizeOutputVector(
    MLAS_INT32X4 IntegerVector,
    MLAS_INT32X4 BiasVector,
    MLAS_FLOAT32X4 ScaleVector,
    MLAS_FLOAT32X4 MinimumValueVector,
    MLAS_FLOAT32X4 MaximumValueVector,
    MLAS_INT32X4 ZeroPointVector
    )
{
    IntegerVector = _mm_add_epi32(IntegerVector, BiasVector);
    MLAS_FLOAT32X4 FloatVector = _mm_cvtepi32_ps(IntegerVector);

    //
    // Scale the input vector and clamp the values to the minimum and maximum
    // range (adjusted by the zero point value).
    //

    FloatVector = MlasMultiplyFloat32x4(FloatVector, ScaleVector);

    // N.B. MINPS and MAXPS returns the value from the second vector if the
    // value from the first vector is a NaN.
    FloatVector = _mm_max_ps(FloatVector, MinimumValueVector);
    FloatVector = _mm_min_ps(FloatVector, MaximumValueVector);

    //
    // Convert the float values to integer using "round to nearest even" and
    // then shift the output range using the zero point value.
    //

    // N.B. Assumes MXCSR has been configured with the default rounding mode of
    // "round to nearest even".
    IntegerVector = _mm_cvtps_epi32(FloatVector);
    IntegerVector = _mm_add_epi32(IntegerVector, ZeroPointVector);

    return IntegerVector;
}

void
MLASCALL
MlasRequantizeOutput(
    const int32_t* Input,
    uint8_t* Output,
    const int32_t* Bias,
    size_t M,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    )
/*++

Routine Description:

    This routine requantizes the intermediate buffer to the output buffer
    optionally adding the supplied bias.

Arguments:

    Input - Supplies the input matrix.

    Output - Supplies the output matrix.

    Bias - Supplies the optional bias vector to be added to the input buffer
        before requantization.

    Buffer - Supplies the output matrix.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    N - Supplies the number of columns of the output matrix.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    MLAS_FLOAT32X4 ScaleVector = MlasBroadcastFloat32x4(Scale);
    MLAS_FLOAT32X4 MinimumValueVector = MlasBroadcastFloat32x4(float(0 - ZeroPoint));
    MLAS_FLOAT32X4 MaximumValueVector = MlasBroadcastFloat32x4(float(255 - ZeroPoint));
    MLAS_INT32X4 ZeroPointVector = MlasBroadcastInt32x4(ZeroPoint);
    MLAS_INT32X4 BiasVector = _mm_setzero_si128();

    //
    // Step through each row of the output matrix.
    //

    while (M-- > 0) {

        if (Bias != nullptr) {
            BiasVector = MlasBroadcastInt32x4(*Bias++);
        }

        size_t n = N;

        while (n >= 4) {

            MLAS_INT32X4 IntegerVector = _mm_loadu_si128((const __m128i *)Input);
            IntegerVector = MlasRequantizeOutputVector(IntegerVector, BiasVector,
                ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVector);

            IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);
            IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);

            *((int32_t*)Output) = _mm_cvtsi128_si32(IntegerVector);

            Input += 4;
            Output += 4;
            n -= 4;
        }

        while (n > 0) {

            MLAS_INT32X4 IntegerVector = _mm_cvtsi32_si128(*Input);
            IntegerVector = MlasRequantizeOutputVector(IntegerVector, BiasVector,
                ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVector);

            *Output = (uint8_t)_mm_cvtsi128_si32(IntegerVector);

            Input += 1;
            Output += 1;
            n -= 1;
        }
    }
}

void
MLASCALL
MlasRequantizeOutputColumn(
    const int32_t* Input,
    uint8_t* Output,
    const int32_t* Bias,
    size_t M,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    )
/*++

Routine Description:

    This routine requantizes the intermediate buffer to the output buffer
    optionally adding the supplied bias.

Arguments:

    Input - Supplies the input matrix.

    Output - Supplies the output matrix.

    Bias - Supplies the optional bias vector to be added to the input buffer
        before requantization.

    Buffer - Supplies the output matrix.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    N - Supplies the number of columns of the output matrix.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    MLAS_FLOAT32X4 ScaleVector = MlasBroadcastFloat32x4(Scale);
    MLAS_FLOAT32X4 MinimumValueVector = MlasBroadcastFloat32x4(float(0 - ZeroPoint));
    MLAS_FLOAT32X4 MaximumValueVector = MlasBroadcastFloat32x4(float(255 - ZeroPoint));
    MLAS_INT32X4 ZeroPointVector = MlasBroadcastInt32x4(ZeroPoint);
    MLAS_INT32X4 BiasVector = _mm_setzero_si128();

    //
    // Step through each row of the output matrix.
    //

    while (M-- > 0) {

        const int32_t* bias = Bias;

        size_t n = N;

        while (n >= 4) {

            MLAS_INT32X4 IntegerVector = _mm_loadu_si128((const __m128i *)Input);

            if (bias != nullptr) {
                BiasVector = _mm_loadu_si128((const __m128i*)bias);
                bias += 4;
            }

            IntegerVector = MlasRequantizeOutputVector(IntegerVector, BiasVector,
                ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVector);

            IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);
            IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);

            *((int32_t*)Output) = _mm_cvtsi128_si32(IntegerVector);

            Input += 4;
            Output += 4;
            n -= 4;
        }

        while (n > 0) {

            MLAS_INT32X4 IntegerVector = _mm_cvtsi32_si128(*Input);

            if (bias != nullptr) {
                BiasVector = _mm_cvtsi32_si128(*bias);
                bias += 1;
            }

            IntegerVector = MlasRequantizeOutputVector(IntegerVector, BiasVector,
                ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVector);

            *Output = (uint8_t)_mm_cvtsi128_si32(IntegerVector);

            Input += 1;
            Output += 1;
            n -= 1;
        }
    }
}

void
MLASCALL
MlasRequantizeOutputColumn(
    const int32_t* Input,
    uint8_t* Output,
    const int32_t* Bias,
    size_t M,
    size_t N,
    const float* Scale,
    uint8_t ZeroPoint
    )
/*++

Routine Description:

    This routine requantizes the intermediate buffer to the output buffer
    optionally adding the supplied bias.

Arguments:

    Input - Supplies the input matrix.

    Output - Supplies the output matrix.

    Bias - Supplies the optional bias vector to be added to the input buffer
        before requantization.

    Buffer - Supplies the output matrix.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    N - Supplies the number of columns of the output matrix.

    Scale - Supplies the quantization scale vector.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    MLAS_FLOAT32X4 MinimumValueVector = MlasBroadcastFloat32x4(float(0 - ZeroPoint));
    MLAS_FLOAT32X4 MaximumValueVector = MlasBroadcastFloat32x4(float(255 - ZeroPoint));
    MLAS_INT32X4 ZeroPointVector = MlasBroadcastInt32x4(ZeroPoint);
    MLAS_INT32X4 BiasVector = _mm_setzero_si128();

    //
    // Step through each row of the output matrix.
    //

    while (M-- > 0) {

        const int32_t* bias = Bias;
        const float* scale = Scale;

        size_t n = N;

        while (n >= 4) {

            MLAS_INT32X4 IntegerVector = _mm_loadu_si128((const __m128i *)Input);

            if (bias != nullptr) {
                BiasVector = _mm_loadu_si128((const __m128i*)bias);
                bias += 4;
            }

            MLAS_FLOAT32X4 ScaleVector = MlasLoadFloat32x4(scale);
            scale += 4;

            IntegerVector = MlasRequantizeOutputVector(IntegerVector, BiasVector,
                ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVector);

            IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);
            IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);

            *((int32_t*)Output) = _mm_cvtsi128_si32(IntegerVector);

            Input += 4;
            Output += 4;
            n -= 4;
        }

        while (n > 0) {

            MLAS_INT32X4 IntegerVector = _mm_cvtsi32_si128(*Input);

            if (bias != nullptr) {
                BiasVector = _mm_cvtsi32_si128(*bias);
                bias += 1;
            }

            MLAS_FLOAT32X4 ScaleVector = _mm_load_ss(scale);
            scale += 1;

            IntegerVector = MlasRequantizeOutputVector(IntegerVector, BiasVector,
                ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVector);

            *Output = (uint8_t)_mm_cvtsi128_si32(IntegerVector);

            Input += 1;
            Output += 1;
            n -= 1;
        }
    }
}

void
MLASCALL
MlasReduceSumU8(
    const uint8_t* Input,
    int32_t* Output,
    size_t Len
    ) {
    *Output = 0;
    if (Len >= 8) {
        const __m128i vzero = _mm_setzero_si128();
        __m128i vacc_lo = _mm_setzero_si128();
        __m128i vacc_hi = _mm_setzero_si128();
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
        __m128i vacc = _mm_add_epi32(vacc_lo, vacc_hi);                    // [ D C | B A ]
        __m128i vshuf = _mm_shuffle_epi32(vacc, _MM_SHUFFLE(2, 3, 0, 1));  // [ C D | A B ]
        __m128i vsums = _mm_add_epi32(vacc, vshuf);                        // [ D+C C+D | B+A A+B ]
        vshuf = _mm_shuffle_epi32(vsums, _MM_SHUFFLE(1, 0, 3, 2));         // [ B+A A+B | D+C C+D ]
        vsums = _mm_add_epi32(vsums, vshuf);
        *Output = _mm_cvtsi128_si32(vsums);
    }
    if (Len > 0) {
        int32_t sum = 0;
        for (; Len > 0; Len--) {
            sum += (int32_t)*Input++;
        }
        *Output += sum;
    }
}

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
    ) {
    const auto vscale = MlasBroadcastFloat32x4(ScaleInput / ScaleOutput);
    const auto vbias = MlasBroadcastInt32x4(gsl::narrow_cast<int32_t>(-1 * ZeroPointInput));
    const auto vmin_value = MlasBroadcastFloat32x4(float(0 - ZeroPointOutput));
    const auto vmax_value = MlasBroadcastFloat32x4(float(255 - ZeroPointOutput));
    const auto vzero_point = MlasBroadcastInt32x4(ZeroPointOutput);

    int32_t sum_buffer[4];
    int sum_count = 0;
    for (; Channels > 0; Channels--) {
        MlasReduceSumU8(Input, sum_buffer + sum_count, ImageSize);
        sum_buffer[sum_count] /= (int32_t)ImageSize;
        Input += ImageSize;
        ++sum_count;
        if (sum_count == 4) {
            auto vsum = _mm_load_si128((__m128i*)sum_buffer);
            auto vresult = MlasRequantizeOutputVector(vsum, vbias, vscale, vmin_value, vmax_value, vzero_point);
            vresult = _mm_packus_epi16(vresult, vresult);
            vresult = _mm_packus_epi16(vresult, vresult);
            *((int32_t*)Output) = _mm_cvtsi128_si32(vresult);
            sum_count = 0;
            Output += 4;
        }
    }
    if (sum_count > 0) {
        auto vsum = _mm_load_si128((__m128i*)sum_buffer);
        auto vresult = MlasRequantizeOutputVector(vsum, vbias, vscale, vmin_value, vmax_value, vzero_point);
        for (; sum_count > 0; --sum_count) {
            *Output++ = (uint8_t)_mm_cvtsi128_si32(vresult);
            vresult = _mm_shuffle_epi32(vresult, _MM_SHUFFLE(0, 3, 2, 1));
        }
    }
}

void
MLASCALL
MlasNhwcQLinearGlobalAveragePoolSingleBatch(
    const uint8_t* Input,
    uint8_t* Output,
    const uint8_t* InputEnd,
    int32_t Bias,
    float Scale,
    int32_t ZeroPointOutput,
    int32_t* AccumulateBuffer,
    const uint8_t* ZeroBuffer,
    size_t ImageSize,
    size_t Channels,
    size_t Stride,
    ) {
  assert(Channels > 1);
  assert(Channels <= Stride);

  bool finish_one_pass = false;
  const __m128i vbias = _mm_set1_epi32(Bias);
  const __m128i vzero = _mm_setzero_si128();
  const auto vscale = MlasBroadcastFloat32x4(Scale);
  const auto vmin_value = MlasBroadcastFloat32x4(float(0 - ZeroPointOutput));
  const auto vmax_value = MlasBroadcastFloat32x4(float(255 - ZeroPointOutput));
  const auto vzero_point = MlasBroadcastInt32x4(ZeroPointOutput);

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

#define CACULATE_ACCUMULATE_VECTORS()                                                                \
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
      uint8_t tail_buffer[8];
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
    const uint8_t* i1 = (ImageSize > 1) ? group_start + Stride : ZeroBuffer;
    const uint8_t* i2 = (ImageSize > 2) ? group_start + Stride * 2 : ZeroBuffer;
    const uint8_t* i3 = (ImageSize > 3) ? group_start + Stride * 3 : ZeroBuffer;
    const uint8_t* i4 = (ImageSize > 4) ? group_start + Stride * 4 : ZeroBuffer;
    const uint8_t* i5 = (ImageSize > 5) ? group_start + Stride * 5 : ZeroBuffer;
    const uint8_t* i6 = (ImageSize > 6) ? group_start + Stride * 6 : ZeroBuffer;

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
      uint8_t tail_buffer[8];
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

static
bool
MLasCalculateParametersForGloabalAveragePool(
    size_t ImageSize,
    float ScaleInput,
    int32_t ZeroPointInput,
    float ScaleOutput,
    int32_t& Bias,
    int32_t& Multiplier,
    int32_t& Shift,
    uint64_t& Rounding
    ){
  Bias = - ZeroPointInput * gsl::narrow_cast<int32_t>(ImageSize);
  float scale = ScaleInput / (ScaleOutput * (float)ImageSize);
  if (scale < 0x1.0p-32f || scale >=  256.0f) return false;

  const uint32_t scale_bits = MlasBitsOfFp32(scale);
  const int32_t Multiplier = (int32_t)(scale_bits & 0x007FFFFF | 0x00800000);
  if (Multiplier < 0x00800000 || Multiplier > 0x00FFFFFF) return false;

  // Shift is in [16, 55] range.
  Shift = 127 + 23 - (scale_bits >> 23);
  if (Shift < 16 || Shift > 55) return false;
  Rounding = uint64_t{1} << ((uint32_t)Shift - 1);

  return true;
}

size_t
MLASCALL
MlasQLinearSafePaddingElementCount(
    size_t ElementSize,
    size_t ElementCount
    ) {
  assert(ElementSize == 1 || ElementSize == 2 || ElementSize == 4 || ElementSize == 8);
  return (ElementSize * ElementCount + size_t{63}) & ~size_t{63};
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
    int32_t* AccumulateBuffer,
    const uint8_t* ZeroBuffer,
    size_t ImageSize,
    size_t Channels,
    size_t Stride,
    size_t Batch
    ) {
  if (Stride <= 1) {
    MlasQLinearGlobalAveragePool(Input, ScaleInput, ZeroPointInput, Output, ScaleOutput, ZeroPointOutput, Channels, ImageSize);
    return;
  }

  int32_t Bias;
  int32_t Multiplier;
  int32_t Shift;
  uint64_t Rounding[2];

  bool fast_ok = MLasCalculateParametersForGloabalAveragePool(
      ImageSize, ScaleInput, ZeroPointInput, ScaleOutput,
      Bias, Multiplier, Shift, Rounding[0]);
  Rounding[1] = Rounding[0];
  const uint8_t* InputEnd = Input + (Batch * ImageSize * Stride - Stride + Channels);

  MlasNhwcReduceSumU8(Input, InputEnd, ZeroBuffer, AccumulateBuffer, ImageSize, Channels, Stride, Bias);

  const auto vscale = MlasBroadcastFloat32x4(ScaleInput / ScaleOutput);
  const auto vmin_value = MlasBroadcastFloat32x4(float(0 - ZeroPointOutput));
  const auto vmax_value = MlasBroadcastFloat32x4(float(255 - ZeroPointOutput));
  const auto vzero_point = MlasBroadcastInt32x4(ZeroPointOutput);

  for (; Channels >=4; Channels-=4) {
    AccumulateBuffer[0] /= (int32_t)ImageSize;
    AccumulateBuffer[1] /= (int32_t)ImageSize;
    AccumulateBuffer[2] /= (int32_t)ImageSize;
    AccumulateBuffer[3] /= (int32_t)ImageSize;
    auto vsum = _mm_load_si128((__m128i*)AccumulateBuffer);
    auto vresult = MlasRequantizeOutputVector(vsum, _mm_setzero_si128(), vscale, vmin_value, vmax_value, vzero_point);
    vresult = _mm_packus_epi16(vresult, vresult);
    vresult = _mm_packus_epi16(vresult, vresult);
    *((int32_t*)Output) = _mm_cvtsi128_si32(vresult);
    Output += 4;
    AccumulateBuffer += 4;
  }
  if (Channels > 0) {
    AccumulateBuffer[0] /= (int32_t)ImageSize;
    AccumulateBuffer[1] /= (int32_t)ImageSize;
    AccumulateBuffer[2] /= (int32_t)ImageSize;
    AccumulateBuffer[3] /= (int32_t)ImageSize;
    auto vsum = _mm_load_si128((__m128i*)AccumulateBuffer);
    auto vresult = MlasRequantizeOutputVector(vsum, _mm_setzero_si128(), vscale, vmin_value, vmax_value, vzero_point);
    for (; Channels > 0; --Channels) {
      *Output++ = (uint8_t)_mm_cvtsi128_si32(vresult);
      vresult = _mm_shuffle_epi32(vresult, _MM_SHUFFLE(0, 3, 2, 1));
    }
  }
}
  
#endif

void
MLASCALL
MlasFindMinMaxElement(
    const float* Input,
    float* Min,
    float* Max,
    size_t N
    )
/*++

Routine Description:

    This routine finds the minimum and maximum values of the supplied buffer.

Arguments:

    Input - Supplies the input buffer.

    Min - Returns the minimum value of the supplied buffer.

    Max - Returns the maximum value of the supplied buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
#if defined(MLAS_TARGET_AMD64)
    MlasPlatform.ReduceMinimumMaximumF32Kernel(Input, Min, Max, N);
#else
    MlasReduceMinimumMaximumF32Kernel(Input, Min, Max, N);
#endif
}
