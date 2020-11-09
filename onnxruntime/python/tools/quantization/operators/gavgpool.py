import onnx
from .base_operator import QuantOperatorBase
from ..quant_utils import attribute_to_kwarg, ms_domain, QuantizedValue, QuantizedValueType

class QGlobalAveragePool(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "GlobalAveragePool")

        # If input to this node is not quantized then keep this node
        if node.input[0] not in self.quantizer.quantized_value_map:
            self.quantizer.new_nodes += [node]
            return

        # Create an entry for output quantized value
        quantized_input_value = self.quantizer.quantized_value_map[node.input[0]]
        quantized_output_value = QuantizedValue(
            node.output[0], node.output[0] + "_quantized",
            quantized_input_value.scale_name, quantized_input_value.zp_name,
            QuantizedValueType.Input)  # Seem not fully flexible in current framework
        self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain
        kwargs["nchw"] = 1
        qnode_name = node.name + "_quant" if node.name != "" else ""

        qnode = onnx.helper.make_node(
            "Quantized" + node.op_type, [quantized_input_value.q_name],
            [quantized_output_value.q_name], qnode_name, **kwargs)
        self.quantizer.new_nodes += [qnode]
