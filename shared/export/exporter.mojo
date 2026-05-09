# shared/export/exporter.mojo
"""
ONNX Exporter for ML Odyssey models.

This module provides the main exporter class for converting
ML Odyssey models to ONNX format for deployment.

The exporter is fully implemented in Mojo, with no Python dependencies.
It directly writes valid ONNX protobuf files that can be loaded by
ONNX Runtime, TensorRT, OpenVINO, and other ONNX-compatible runtimes.
"""

from shared.export.protobuf import ProtoBuffer
from shared.export.onnx_proto import (
    ModelProto,
    GraphProto,
    NodeProto,
    ValueInfoProto,
    TensorProto,
    AttributeProto,
    ONNX_FLOAT,
    ONNX_DOUBLE,
    ONNX_INT64,
    dtype_to_onnx,
)


struct ExportConfig(Copyable, Movable):
    """Configuration for ONNX export."""

    var opset_version: Int64
    var ir_version: Int64
    var producer_name: String
    var producer_version: String
    var model_name: String
    var doc_string: String

    def __init__(
        out self,
        opset_version: Int64 = 14,
        model_name: String = "model",
    ):
        """Initialize export configuration.

        Args:
            opset_version: ONNX opset version (default is 14).
            model_name: Name for the exported model.
        """
        self.opset_version = opset_version
        self.ir_version = 9  # IR version 9 for opset 14+
        self.producer_name = String("ML Odyssey")
        self.producer_version = String("1.0.0")
        self.model_name = model_name
        self.doc_string = String("")


struct ONNXExporter(Movable):
    """Export ML Odyssey models to ONNX format."""

    var config: ExportConfig
    var model: ModelProto
    var verbose: Bool

    def __init__(out self, opset_version: Int64 = 14, verbose: Bool = False):
        """Initialize ONNX exporter.

        Args:
            opset_version: ONNX opset version (default is 14).
            verbose: Enable verbose output.
        """
        self.config = ExportConfig(opset_version)
        self.model = ModelProto()
        self.model.set_opset(opset_version)
        self.verbose = verbose

    def set_model_name(mut self, name: String):
        """Set the model name."""
        self.model.graph.name = name

    def set_doc_string(mut self, doc: String):
        """Set model documentation string."""
        self.model.doc_string = doc

    def add_input(
        mut self,
        name: String,
        var shape: List[Int64],
        dtype: String = "float32",
    ):
        """Add an input tensor specification."""
        var input = ValueInfoProto(name, dtype_to_onnx(dtype))
        for i in range(len(shape)):
            input.add_dim(shape[i])
        self.model.graph.add_input(input^)

        if self.verbose:
            print("Added input:", name)

    def add_input_dynamic(
        mut self,
        name: String,
        var shape: List[Int64],
        var dynamic_dims: List[Int],
        var dim_names: List[String],
        dtype: String = "float32",
    ):
        """Add an input with dynamic dimensions."""
        var input = ValueInfoProto(name, dtype_to_onnx(dtype))

        var dyn_idx = 0
        for i in range(len(shape)):
            var is_dynamic = False
            for j in range(len(dynamic_dims)):
                if dynamic_dims[j] == i:
                    is_dynamic = True
                    break

            if is_dynamic and dyn_idx < len(dim_names):
                input.add_dim_param(dim_names[dyn_idx])
                dyn_idx += 1
            else:
                input.add_dim(shape[i])

        self.model.graph.add_input(input^)

    def add_output(
        mut self,
        name: String,
        var shape: List[Int64],
        dtype: String = "float32",
    ):
        """Add an output tensor specification."""
        var output = ValueInfoProto(name, dtype_to_onnx(dtype))
        for i in range(len(shape)):
            output.add_dim(shape[i])
        self.model.graph.add_output(output^)

        if self.verbose:
            print("Added output:", name)

    def add_initializer(
        mut self,
        name: String,
        var shape: List[Int64],
        var data: List[Float32],
    ):
        """Add a float32 initializer (weights/biases)."""
        var tensor = TensorProto(name, ONNX_FLOAT)
        tensor.set_dims(shape^)
        tensor.set_float_data(data^)
        self.model.graph.add_initializer(tensor^)

        if self.verbose:
            print("Added initializer:", name)

    def add_conv(
        mut self,
        name: String,
        input_name: String,
        output_name: String,
        weight_name: String,
        bias_name: String,
        var kernel_shape: List[Int64],
        var strides: List[Int64],
        var pads: List[Int64],
        var dilations: List[Int64],
        groups: Int64 = 1,
    ):
        """Add a Conv node."""
        var node = NodeProto(name, String("Conv"))
        node.add_input(input_name)
        node.add_input(weight_name)
        if bias_name.byte_length() > 0:
            node.add_input(bias_name)
        node.add_output(output_name)

        node.add_ints_attr(String("kernel_shape"), kernel_shape^)
        node.add_ints_attr(String("strides"), strides^)
        node.add_ints_attr(String("pads"), pads^)
        node.add_ints_attr(String("dilations"), dilations^)
        node.add_int_attr(String("group"), groups)

        self.model.graph.add_node(node^)

    def add_gemm(
        mut self,
        name: String,
        input_name: String,
        weight_name: String,
        bias_name: String,
        output_name: String,
        alpha: Float32 = 1.0,
        beta: Float32 = 1.0,
        transA: Int64 = 0,
        transB: Int64 = 1,
    ):
        """Add a Gemm (Linear) node."""
        var node = NodeProto(name, String("Gemm"))
        node.add_input(input_name)
        node.add_input(weight_name)
        node.add_input(bias_name)
        node.add_output(output_name)

        node.add_float_attr(String("alpha"), alpha)
        node.add_float_attr(String("beta"), beta)
        node.add_int_attr(String("transA"), transA)
        node.add_int_attr(String("transB"), transB)

        self.model.graph.add_node(node^)

    def add_batchnorm(
        mut self,
        name: String,
        input_name: String,
        scale_name: String,
        bias_name: String,
        mean_name: String,
        var_name: String,
        output_name: String,
        epsilon: Float32 = 1e-5,
        momentum: Float32 = 0.9,
    ):
        """Add a BatchNormalization node."""
        var node = NodeProto(name, String("BatchNormalization"))
        node.add_input(input_name)
        node.add_input(scale_name)
        node.add_input(bias_name)
        node.add_input(mean_name)
        node.add_input(var_name)
        node.add_output(output_name)

        node.add_float_attr(String("epsilon"), epsilon)
        node.add_float_attr(String("momentum"), momentum)

        self.model.graph.add_node(node^)

    def add_relu(
        mut self,
        name: String,
        input_name: String,
        output_name: String,
    ):
        """Add a ReLU activation node."""
        var node = NodeProto(name, String("Relu"))
        node.add_input(input_name)
        node.add_output(output_name)
        self.model.graph.add_node(node^)

    def add_sigmoid(
        mut self,
        name: String,
        input_name: String,
        output_name: String,
    ):
        """Add a Sigmoid activation node."""
        var node = NodeProto(name, String("Sigmoid"))
        node.add_input(input_name)
        node.add_output(output_name)
        self.model.graph.add_node(node^)

    def add_tanh(
        mut self,
        name: String,
        input_name: String,
        output_name: String,
    ):
        """Add a Tanh activation node."""
        var node = NodeProto(name, String("Tanh"))
        node.add_input(input_name)
        node.add_output(output_name)
        self.model.graph.add_node(node^)

    def add_softmax(
        mut self,
        name: String,
        input_name: String,
        output_name: String,
        axis: Int64 = -1,
    ):
        """Add a Softmax node."""
        var node = NodeProto(name, String("Softmax"))
        node.add_input(input_name)
        node.add_output(output_name)
        node.add_int_attr(String("axis"), axis)
        self.model.graph.add_node(node^)

    def add_maxpool(
        mut self,
        name: String,
        input_name: String,
        output_name: String,
        var kernel_shape: List[Int64],
        var strides: List[Int64],
        var pads: List[Int64],
    ):
        """Add a MaxPool node."""
        var node = NodeProto(name, String("MaxPool"))
        node.add_input(input_name)
        node.add_output(output_name)

        node.add_ints_attr(String("kernel_shape"), kernel_shape^)
        node.add_ints_attr(String("strides"), strides^)
        node.add_ints_attr(String("pads"), pads^)

        self.model.graph.add_node(node^)

    def add_avgpool(
        mut self,
        name: String,
        input_name: String,
        output_name: String,
        var kernel_shape: List[Int64],
        var strides: List[Int64],
        var pads: List[Int64],
    ):
        """Add an AveragePool node."""
        var node = NodeProto(name, String("AveragePool"))
        node.add_input(input_name)
        node.add_output(output_name)

        node.add_ints_attr(String("kernel_shape"), kernel_shape^)
        node.add_ints_attr(String("strides"), strides^)
        node.add_ints_attr(String("pads"), pads^)

        self.model.graph.add_node(node^)

    def add_global_avgpool(
        mut self,
        name: String,
        input_name: String,
        output_name: String,
    ):
        """Add a GlobalAveragePool node."""
        var node = NodeProto(name, String("GlobalAveragePool"))
        node.add_input(input_name)
        node.add_output(output_name)
        self.model.graph.add_node(node^)

    def add_flatten(
        mut self,
        name: String,
        input_name: String,
        output_name: String,
        axis: Int64 = 1,
    ):
        """Add a Flatten node."""
        var node = NodeProto(name, String("Flatten"))
        node.add_input(input_name)
        node.add_output(output_name)
        node.add_int_attr(String("axis"), axis)
        self.model.graph.add_node(node^)

    def add_dropout(
        mut self,
        name: String,
        input_name: String,
        output_name: String,
        ratio: Float32 = 0.5,
    ):
        """Add a Dropout node (identity in inference mode)."""
        var node = NodeProto(name, String("Dropout"))
        node.add_input(input_name)
        node.add_output(output_name)
        node.add_float_attr(String("ratio"), ratio)
        self.model.graph.add_node(node^)

    def add_add(
        mut self,
        name: String,
        input_a: String,
        input_b: String,
        output_name: String,
    ):
        """Add an element-wise Add node."""
        var node = NodeProto(name, String("Add"))
        node.add_input(input_a)
        node.add_input(input_b)
        node.add_output(output_name)
        self.model.graph.add_node(node^)

    def add_concat(
        mut self,
        name: String,
        var inputs: List[String],
        output_name: String,
        axis: Int64 = 1,
    ):
        """Add a Concat node."""
        var node = NodeProto(name, String("Concat"))
        for i in range(len(inputs)):
            node.add_input(inputs[i])
        node.add_output(output_name)
        node.add_int_attr(String("axis"), axis)
        self.model.graph.add_node(node^)

    def add_reshape(
        mut self,
        name: String,
        input_name: String,
        shape_name: String,
        output_name: String,
    ):
        """Add a Reshape node."""
        var node = NodeProto(name, String("Reshape"))
        node.add_input(input_name)
        node.add_input(shape_name)
        node.add_output(output_name)
        self.model.graph.add_node(node^)

    def add_transpose(
        mut self,
        name: String,
        input_name: String,
        output_name: String,
        var perm: List[Int64],
    ):
        """Add a Transpose node."""
        var node = NodeProto(name, String("Transpose"))
        node.add_input(input_name)
        node.add_output(output_name)
        node.add_ints_attr(String("perm"), perm^)
        self.model.graph.add_node(node^)

    def add_matmul(
        mut self,
        name: String,
        input_a: String,
        input_b: String,
        output_name: String,
    ):
        """Add a MatMul node."""
        var node = NodeProto(name, String("MatMul"))
        node.add_input(input_a)
        node.add_input(input_b)
        node.add_output(output_name)
        self.model.graph.add_node(node^)

    def num_nodes(self) -> Int:
        """Get the number of nodes in the graph."""
        return len(self.model.graph.nodes)

    def num_initializers(self) -> Int:
        """Get the number of initializers."""
        return len(self.model.graph.initializers)

    def save(self, path: String) raises:
        """Save the model to an ONNX file.

        Args:
            path: Output file path.

        Raises:
            Error if file cannot be written.
        """
        if self.verbose:
            print("Saving ONNX model to:", path)
            print("  Nodes:", self.num_nodes())
            print("  Initializers:", self.num_initializers())
            print("  Opset:", self.config.opset_version)

        # Encode model to protobuf
        var buf = self.model.encode()

        # Write to file
        var bytes = buf.get_bytes()

        # Use file I/O to write binary data
        with open(path, "wb") as f:
            for i in range(len(bytes)):
                # Write each byte
                f.write(String(chr(Int(bytes[i]))))

        if self.verbose:
            print("Model saved successfully!")
            print("  File size:", len(bytes), "bytes")


def create_lenet5_onnx() -> ONNXExporter:
    """Create a LeNet-5 ONNX model for demonstration.

    Returns:
        ONNXExporter configured with LeNet-5 architecture.
    """
    var exporter = ONNXExporter(opset_version=14, verbose=True)
    exporter.set_model_name(String("LeNet5"))
    exporter.set_doc_string(String("LeNet-5 MNIST classifier"))

    # Input: [batch, 1, 28, 28]
    var input_shape = List[Int64]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(28)
    input_shape.append(28)
    exporter.add_input(String("input"), input_shape^)

    # Conv1: 1 -> 6 channels, 5x5 kernel, padding 2
    var k1 = List[Int64]()
    k1.append(5)
    k1.append(5)
    var s1 = List[Int64]()
    s1.append(1)
    s1.append(1)
    var p1 = List[Int64]()
    p1.append(2)
    p1.append(2)
    p1.append(2)
    p1.append(2)
    var d1 = List[Int64]()
    d1.append(1)
    d1.append(1)

    exporter.add_conv(
        String("conv1"),
        String("input"),
        String("conv1_out"),
        String("conv1.weight"),
        String("conv1.bias"),
        k1^,
        s1^,
        p1^,
        d1^,
    )

    # ReLU1
    exporter.add_relu(String("relu1"), String("conv1_out"), String("relu1_out"))

    # MaxPool1: 2x2
    var k2 = List[Int64]()
    k2.append(2)
    k2.append(2)
    var s2 = List[Int64]()
    s2.append(2)
    s2.append(2)
    var p2 = List[Int64]()
    p2.append(0)
    p2.append(0)
    p2.append(0)
    p2.append(0)

    exporter.add_maxpool(
        String("pool1"), String("relu1_out"), String("pool1_out"), k2^, s2^, p2^
    )

    # Conv2: 6 -> 16 channels, 5x5 kernel
    var k3 = List[Int64]()
    k3.append(5)
    k3.append(5)
    var s3 = List[Int64]()
    s3.append(1)
    s3.append(1)
    var p3 = List[Int64]()
    p3.append(0)
    p3.append(0)
    p3.append(0)
    p3.append(0)
    var d3 = List[Int64]()
    d3.append(1)
    d3.append(1)

    exporter.add_conv(
        String("conv2"),
        String("pool1_out"),
        String("conv2_out"),
        String("conv2.weight"),
        String("conv2.bias"),
        k3^,
        s3^,
        p3^,
        d3^,
    )

    # ReLU2
    exporter.add_relu(String("relu2"), String("conv2_out"), String("relu2_out"))

    # MaxPool2: 2x2
    var k4 = List[Int64]()
    k4.append(2)
    k4.append(2)
    var s4 = List[Int64]()
    s4.append(2)
    s4.append(2)
    var p4 = List[Int64]()
    p4.append(0)
    p4.append(0)
    p4.append(0)
    p4.append(0)

    exporter.add_maxpool(
        String("pool2"), String("relu2_out"), String("pool2_out"), k4^, s4^, p4^
    )

    # Flatten
    exporter.add_flatten(
        String("flatten"), String("pool2_out"), String("flat_out")
    )

    # FC1: 400 -> 120
    exporter.add_gemm(
        String("fc1"),
        String("flat_out"),
        String("fc1.weight"),
        String("fc1.bias"),
        String("fc1_out"),
    )

    # ReLU3
    exporter.add_relu(String("relu3"), String("fc1_out"), String("relu3_out"))

    # FC2: 120 -> 84
    exporter.add_gemm(
        String("fc2"),
        String("relu3_out"),
        String("fc2.weight"),
        String("fc2.bias"),
        String("fc2_out"),
    )

    # ReLU4
    exporter.add_relu(String("relu4"), String("fc2_out"), String("relu4_out"))

    # FC3: 84 -> 10
    exporter.add_gemm(
        String("fc3"),
        String("relu4_out"),
        String("fc3.weight"),
        String("fc3.bias"),
        String("output"),
    )

    # Output: [batch, 10]
    var output_shape = List[Int64]()
    output_shape.append(1)
    output_shape.append(10)
    exporter.add_output(String("output"), output_shape^)

    return exporter^
