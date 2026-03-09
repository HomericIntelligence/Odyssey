# shared/export/onnx_proto.mojo
"""
ONNX Protocol Buffer message definitions.

This module implements the ONNX protobuf schema for serializing
ML Odyssey models to the ONNX format.

Based on ONNX IR specification v9 (opset 14+).
Reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
"""

from shared.export.protobuf import ProtoBuffer


# ONNX data types.
alias ONNX_UNDEFINED: Int = 0
alias ONNX_FLOAT: Int = 1
alias ONNX_UINT8: Int = 2
alias ONNX_INT8: Int = 3
alias ONNX_UINT16: Int = 4
alias ONNX_INT16: Int = 5
alias ONNX_INT32: Int = 6
alias ONNX_INT64: Int = 7
alias ONNX_STRING: Int = 8
alias ONNX_BOOL: Int = 9
alias ONNX_FLOAT16: Int = 10
alias ONNX_DOUBLE: Int = 11
alias ONNX_UINT32: Int = 12
alias ONNX_UINT64: Int = 13
alias ONNX_COMPLEX64: Int = 14
alias ONNX_COMPLEX128: Int = 15
alias ONNX_BFLOAT16: Int = 16


fn dtype_to_onnx(dtype: String) -> Int:
    """Convert dtype string to ONNX data type.

    Args:
        dtype: Data type string such as `float32` or `int64`.

    Returns:
        ONNX data type constant.
    """
    if dtype == "float32":
        return ONNX_FLOAT
    elif dtype == "float64":
        return ONNX_DOUBLE
    elif dtype == "float16":
        return ONNX_FLOAT16
    elif dtype == "bfloat16":
        return ONNX_BFLOAT16
    elif dtype == "int32":
        return ONNX_INT32
    elif dtype == "int64":
        return ONNX_INT64
    elif dtype == "int8":
        return ONNX_INT8
    elif dtype == "uint8":
        return ONNX_UINT8
    elif dtype == "bool":
        return ONNX_BOOL
    else:
        return ONNX_FLOAT  # Default


# AttributeProto.AttributeType.
alias ATTR_UNDEFINED: Int = 0
alias ATTR_FLOAT: Int = 1
alias ATTR_INT: Int = 2
alias ATTR_STRING: Int = 3
alias ATTR_TENSOR: Int = 4
alias ATTR_GRAPH: Int = 5
alias ATTR_FLOATS: Int = 6
alias ATTR_INTS: Int = 7
alias ATTR_STRINGS: Int = 8


struct AttributeProto(Copyable, Movable):
    """ONNX AttributeProto message.

    Represents an attribute of a node such as `kernel_size` or `stride`.
    """

    var name: String
    var attr_type: Int
    var f: Float32  # float value
    var i: Int64  # int value
    var s: String  # string value
    var floats: List[Float32]  # repeated float
    var ints: List[Int64]  # repeated int

    fn __init__(out self, name: String):
        """Initialize attribute with name."""
        self.name = name
        self.attr_type = ATTR_UNDEFINED
        self.f = 0.0
        self.i = 0
        self.s = String("")
        self.floats = List[Float32]()
        self.ints = List[Int64]()

    fn copy(self) -> Self:
        """Create a copy of this attribute."""
        var result = AttributeProto(self.name)
        result.attr_type = self.attr_type
        result.f = self.f
        result.i = self.i
        result.s = self.s
        for j in range(len(self.floats)):
            result.floats.append(self.floats[j])
        for j in range(len(self.ints)):
            result.ints.append(self.ints[j])
        return result^

    fn __copyinit__(out self, read other: Self):
        """Copy constructor."""
        self.name = other.name
        self.attr_type = other.attr_type
        self.f = other.f
        self.i = other.i
        self.s = other.s
        self.floats = List[Float32]()
        for j in range(len(other.floats)):
            self.floats.append(other.floats[j])
        self.ints = List[Int64]()
        for j in range(len(other.ints)):
            self.ints.append(other.ints[j])

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.name = other.name^
        self.attr_type = other.attr_type
        self.f = other.f
        self.i = other.i
        self.s = other.s^
        self.floats = other.floats^
        self.ints = other.ints^

    fn set_float(mut self, value: Float32):
        """Set float value."""
        self.attr_type = ATTR_FLOAT
        self.f = value

    fn set_int(mut self, value: Int64):
        """Set int value."""
        self.attr_type = ATTR_INT
        self.i = value

    fn set_string(mut self, value: String):
        """Set string value."""
        self.attr_type = ATTR_STRING
        self.s = value

    fn set_floats(mut self, var values: List[Float32]):
        """Set repeated float values."""
        self.attr_type = ATTR_FLOATS
        self.floats = values^

    fn set_ints(mut self, var values: List[Int64]):
        """Set repeated int values."""
        self.attr_type = ATTR_INTS
        self.ints = values^

    fn encode(self) -> ProtoBuffer:
        """Encode to protobuf bytes."""
        var buf = ProtoBuffer()
        buf.write_string(1, self.name)
        buf.write_int32(4, Int32(self.attr_type))

        if self.attr_type == ATTR_FLOAT:
            buf.write_float(5, self.f)
        elif self.attr_type == ATTR_INT:
            buf.write_int64(6, self.i)
        elif self.attr_type == ATTR_STRING:
            buf.write_string(7, self.s)
        elif self.attr_type == ATTR_FLOATS:
            buf.write_packed_float(21, self.floats)
        elif self.attr_type == ATTR_INTS:
            buf.write_packed_int64(22, self.ints)

        return buf^


struct TensorShapeProto(Copyable, Movable):
    """ONNX TensorShapeProto message."""

    var dims: List[Int64]
    var dim_params: List[String]

    fn __init__(out self):
        self.dims = List[Int64]()
        self.dim_params = List[String]()

    fn __copyinit__(out self, read other: Self):
        self.dims = List[Int64]()
        for j in range(len(other.dims)):
            self.dims.append(other.dims[j])
        self.dim_params = List[String]()
        for j in range(len(other.dim_params)):
            self.dim_params.append(other.dim_params[j])

    fn __moveinit__(out self, deinit other: Self):
        self.dims = other.dims^
        self.dim_params = other.dim_params^

    fn add_dim(mut self, size: Int64):
        """Add a dimension with fixed size."""
        self.dims.append(size)
        self.dim_params.append(String(""))

    fn add_dim_param(mut self, name: String):
        """Add a dimension with symbolic name (dynamic)."""
        self.dims.append(-1)
        self.dim_params.append(name)

    fn encode(self) -> ProtoBuffer:
        """Encode to protobuf bytes."""
        var buf = ProtoBuffer()
        for i in range(len(self.dims)):
            var dim_buf = ProtoBuffer()
            if len(self.dim_params[i]) > 0:
                dim_buf.write_string(2, self.dim_params[i])
            else:
                dim_buf.write_int64(1, self.dims[i])
            buf.write_embedded_message(1, dim_buf)
        return buf^


struct TypeProto(Copyable, Movable):
    """ONNX TypeProto message."""

    var elem_type: Int
    var shape: TensorShapeProto

    fn __init__(out self, elem_type: Int = ONNX_FLOAT):
        self.elem_type = elem_type
        self.shape = TensorShapeProto()

    fn __copyinit__(out self, read other: Self):
        self.elem_type = other.elem_type
        self.shape = TensorShapeProto()
        for j in range(len(other.shape.dims)):
            self.shape.dims.append(other.shape.dims[j])
        for j in range(len(other.shape.dim_params)):
            self.shape.dim_params.append(other.shape.dim_params[j])

    fn __moveinit__(out self, deinit other: Self):
        self.elem_type = other.elem_type
        self.shape = other.shape^

    fn encode(self) -> ProtoBuffer:
        """Encode to protobuf bytes."""
        var tensor_buf = ProtoBuffer()
        tensor_buf.write_int32(1, Int32(self.elem_type))
        var shape_buf = self.shape.encode()
        tensor_buf.write_embedded_message(2, shape_buf)

        var buf = ProtoBuffer()
        buf.write_embedded_message(1, tensor_buf)
        return buf^


struct ValueInfoProto(Copyable, Movable):
    """ONNX ValueInfoProto message."""

    var name: String
    var type_proto: TypeProto
    var doc_string: String

    fn __init__(out self, name: String, elem_type: Int = ONNX_FLOAT):
        self.name = name
        self.type_proto = TypeProto(elem_type)
        self.doc_string = String("")

    fn __copyinit__(out self, read other: Self):
        self.name = other.name
        self.type_proto = TypeProto(other.type_proto.elem_type)
        for j in range(len(other.type_proto.shape.dims)):
            self.type_proto.shape.dims.append(other.type_proto.shape.dims[j])
        for j in range(len(other.type_proto.shape.dim_params)):
            self.type_proto.shape.dim_params.append(
                other.type_proto.shape.dim_params[j]
            )
        self.doc_string = other.doc_string

    fn __moveinit__(out self, deinit other: Self):
        self.name = other.name^
        self.type_proto = other.type_proto^
        self.doc_string = other.doc_string^

    fn add_dim(mut self, size: Int64):
        """Add a dimension to the shape."""
        self.type_proto.shape.add_dim(size)

    fn add_dim_param(mut self, name: String):
        """Add a dynamic dimension."""
        self.type_proto.shape.add_dim_param(name)

    fn encode(self) -> ProtoBuffer:
        """Encode to protobuf bytes."""
        var buf = ProtoBuffer()
        buf.write_string(1, self.name)
        var type_buf = self.type_proto.encode()
        buf.write_embedded_message(2, type_buf)
        if len(self.doc_string) > 0:
            buf.write_string(3, self.doc_string)
        return buf^


struct TensorProto(Copyable, Movable):
    """ONNX TensorProto message."""

    var name: String
    var data_type: Int
    var dims: List[Int64]
    var float_data: List[Float32]
    var int64_data: List[Int64]
    var raw_data: List[UInt8]

    fn __init__(out self, name: String, data_type: Int = ONNX_FLOAT):
        self.name = name
        self.data_type = data_type
        self.dims = List[Int64]()
        self.float_data = List[Float32]()
        self.int64_data = List[Int64]()
        self.raw_data = List[UInt8]()

    fn __copyinit__(out self, read other: Self):
        self.name = other.name
        self.data_type = other.data_type
        self.dims = List[Int64]()
        for j in range(len(other.dims)):
            self.dims.append(other.dims[j])
        self.float_data = List[Float32]()
        for j in range(len(other.float_data)):
            self.float_data.append(other.float_data[j])
        self.int64_data = List[Int64]()
        for j in range(len(other.int64_data)):
            self.int64_data.append(other.int64_data[j])
        self.raw_data = List[UInt8]()
        for j in range(len(other.raw_data)):
            self.raw_data.append(other.raw_data[j])

    fn __moveinit__(out self, deinit other: Self):
        self.name = other.name^
        self.data_type = other.data_type
        self.dims = other.dims^
        self.float_data = other.float_data^
        self.int64_data = other.int64_data^
        self.raw_data = other.raw_data^

    fn set_dims(mut self, var dims: List[Int64]):
        """Set tensor dimensions."""
        self.dims = dims^

    fn set_float_data(mut self, var data: List[Float32]):
        """Set float data."""
        self.float_data = data^

    fn set_raw_data(mut self, var data: List[UInt8]):
        """Set raw binary data."""
        self.raw_data = data^

    fn encode(self) -> ProtoBuffer:
        """Encode to protobuf bytes."""
        var buf = ProtoBuffer()
        if len(self.dims) > 0:
            buf.write_packed_int64(1, self.dims)
        buf.write_int32(2, Int32(self.data_type))
        buf.write_string(8, self.name)

        if len(self.raw_data) > 0:
            buf.write_bytes_field(9, self.raw_data)
        elif len(self.float_data) > 0:
            buf.write_packed_float(4, self.float_data)
        elif len(self.int64_data) > 0:
            buf.write_packed_int64(7, self.int64_data)

        return buf^


struct NodeProto(Copyable, Movable):
    """ONNX NodeProto message."""

    var name: String
    var op_type: String
    var inputs: List[String]
    var outputs: List[String]
    var attributes: List[AttributeProto]
    var domain: String
    var doc_string: String

    fn __init__(out self, name: String, op_type: String):
        self.name = name
        self.op_type = op_type
        self.inputs = List[String]()
        self.outputs = List[String]()
        self.attributes = List[AttributeProto]()
        self.domain = String("")
        self.doc_string = String("")

    fn __copyinit__(out self, read other: Self):
        self.name = other.name
        self.op_type = other.op_type
        self.inputs = List[String]()
        for j in range(len(other.inputs)):
            self.inputs.append(other.inputs[j])
        self.outputs = List[String]()
        for j in range(len(other.outputs)):
            self.outputs.append(other.outputs[j])
        self.attributes = List[AttributeProto]()
        for j in range(len(other.attributes)):
            self.attributes.append(other.attributes[j].copy())
        self.domain = other.domain
        self.doc_string = other.doc_string

    fn __moveinit__(out self, deinit other: Self):
        self.name = other.name^
        self.op_type = other.op_type^
        self.inputs = other.inputs^
        self.outputs = other.outputs^
        self.attributes = other.attributes^
        self.domain = other.domain^
        self.doc_string = other.doc_string^

    fn add_input(mut self, name: String):
        """Add an input tensor name."""
        self.inputs.append(name)

    fn add_output(mut self, name: String):
        """Add an output tensor name."""
        self.outputs.append(name)

    fn add_attribute(mut self, var attr: AttributeProto):
        """Add an attribute."""
        self.attributes.append(attr^)

    fn add_int_attr(mut self, name: String, value: Int64):
        """Add an integer attribute."""
        var attr = AttributeProto(name)
        attr.set_int(value)
        self.attributes.append(attr^)

    fn add_float_attr(mut self, name: String, value: Float32):
        """Add a float attribute."""
        var attr = AttributeProto(name)
        attr.set_float(value)
        self.attributes.append(attr^)

    fn add_ints_attr(mut self, name: String, var values: List[Int64]):
        """Add an integer list attribute."""
        var attr = AttributeProto(name)
        attr.set_ints(values^)
        self.attributes.append(attr^)

    fn encode(self) -> ProtoBuffer:
        """Encode to protobuf bytes."""
        var buf = ProtoBuffer()

        for i in range(len(self.inputs)):
            buf.write_string(1, self.inputs[i])
        for i in range(len(self.outputs)):
            buf.write_string(2, self.outputs[i])

        buf.write_string(3, self.name)
        buf.write_string(4, self.op_type)

        for i in range(len(self.attributes)):
            var attr_buf = self.attributes[i].encode()
            buf.write_embedded_message(5, attr_buf)

        if len(self.doc_string) > 0:
            buf.write_string(6, self.doc_string)
        if len(self.domain) > 0:
            buf.write_string(7, self.domain)

        return buf^


struct OperatorSetIdProto(Copyable, Movable):
    """ONNX OperatorSetIdProto message."""

    var domain: String
    var version: Int64

    fn __init__(out self, domain: String = "", version: Int64 = 14):
        self.domain = domain
        self.version = version

    fn __copyinit__(out self, read other: Self):
        self.domain = other.domain
        self.version = other.version

    fn __moveinit__(out self, deinit other: Self):
        self.domain = other.domain^
        self.version = other.version

    fn encode(self) -> ProtoBuffer:
        """Encode to protobuf bytes."""
        var buf = ProtoBuffer()
        if len(self.domain) > 0:
            buf.write_string(1, self.domain)
        buf.write_int64(2, self.version)
        return buf^


struct GraphProto(Copyable, Movable):
    """ONNX GraphProto message."""

    var name: String
    var nodes: List[NodeProto]
    var inputs: List[ValueInfoProto]
    var outputs: List[ValueInfoProto]
    var initializers: List[TensorProto]
    var doc_string: String

    fn __init__(out self, name: String = "graph"):
        self.name = name
        self.nodes = List[NodeProto]()
        self.inputs = List[ValueInfoProto]()
        self.outputs = List[ValueInfoProto]()
        self.initializers = List[TensorProto]()
        self.doc_string = String("")

    fn __copyinit__(out self, read other: Self):
        self.name = other.name
        self.nodes = List[NodeProto]()
        self.inputs = List[ValueInfoProto]()
        self.outputs = List[ValueInfoProto]()
        self.initializers = List[TensorProto]()
        self.doc_string = other.doc_string
        # Deep copy would be complex - keeping empty for now

    fn __moveinit__(out self, deinit other: Self):
        self.name = other.name^
        self.nodes = other.nodes^
        self.inputs = other.inputs^
        self.outputs = other.outputs^
        self.initializers = other.initializers^
        self.doc_string = other.doc_string^

    fn add_node(mut self, var node: NodeProto):
        """Add a node to the graph."""
        self.nodes.append(node^)

    fn add_input(mut self, var input: ValueInfoProto):
        """Add an input specification."""
        self.inputs.append(input^)

    fn add_output(mut self, var output: ValueInfoProto):
        """Add an output specification."""
        self.outputs.append(output^)

    fn add_initializer(mut self, var tensor: TensorProto):
        """Add an initializer (weight/bias)."""
        self.initializers.append(tensor^)

    fn encode(self) -> ProtoBuffer:
        """Encode to protobuf bytes."""
        var buf = ProtoBuffer()

        for i in range(len(self.nodes)):
            var node_buf = self.nodes[i].encode()
            buf.write_embedded_message(1, node_buf)

        buf.write_string(2, self.name)

        for i in range(len(self.initializers)):
            var init_buf = self.initializers[i].encode()
            buf.write_embedded_message(3, init_buf)

        if len(self.doc_string) > 0:
            buf.write_string(5, self.doc_string)

        for i in range(len(self.inputs)):
            var input_buf = self.inputs[i].encode()
            buf.write_embedded_message(6, input_buf)

        for i in range(len(self.outputs)):
            var output_buf = self.outputs[i].encode()
            buf.write_embedded_message(7, output_buf)

        return buf^


struct ModelProto(Copyable, Movable):
    """ONNX ModelProto message."""

    var ir_version: Int64
    var producer_name: String
    var producer_version: String
    var domain: String
    var model_version: Int64
    var doc_string: String
    var graph: GraphProto
    var opset_imports: List[OperatorSetIdProto]

    fn __init__(out self, ir_version: Int64 = 9):
        """Initialize model with IR version.

        Args:
            ir_version: ONNX IR version (default is 9 for opset 14+).
        """
        self.ir_version = ir_version
        self.producer_name = String("ML Odyssey")
        self.producer_version = String("1.0.0")
        self.domain = String("")
        self.model_version = 1
        self.doc_string = String("")
        self.graph = GraphProto()
        self.opset_imports = List[OperatorSetIdProto]()

    fn __copyinit__(out self, read other: Self):
        self.ir_version = other.ir_version
        self.producer_name = other.producer_name
        self.producer_version = other.producer_version
        self.domain = other.domain
        self.model_version = other.model_version
        self.doc_string = other.doc_string
        self.graph = GraphProto(other.graph.name)
        self.opset_imports = List[OperatorSetIdProto]()
        for j in range(len(other.opset_imports)):
            self.opset_imports.append(
                OperatorSetIdProto(
                    other.opset_imports[j].domain,
                    other.opset_imports[j].version,
                )
            )

    fn __moveinit__(out self, deinit other: Self):
        self.ir_version = other.ir_version
        self.producer_name = other.producer_name^
        self.producer_version = other.producer_version^
        self.domain = other.domain^
        self.model_version = other.model_version
        self.doc_string = other.doc_string^
        self.graph = other.graph^
        self.opset_imports = other.opset_imports^

    fn set_opset(mut self, version: Int64, domain: String = ""):
        """Set the opset version.

        Args:
            version: Opset version number.
            domain: Operator domain (empty for default ONNX domain).
        """
        self.opset_imports.append(OperatorSetIdProto(domain, version))

    fn encode(self) -> ProtoBuffer:
        """Encode to protobuf bytes."""
        var buf = ProtoBuffer()

        buf.write_int64(1, self.ir_version)

        for i in range(len(self.opset_imports)):
            var opset_buf = self.opset_imports[i].encode()
            buf.write_embedded_message(2, opset_buf)

        buf.write_string(3, self.producer_name)
        buf.write_string(4, self.producer_version)

        if len(self.domain) > 0:
            buf.write_string(5, self.domain)

        buf.write_int64(6, self.model_version)

        if len(self.doc_string) > 0:
            buf.write_string(7, self.doc_string)

        var graph_buf = self.graph.encode()
        buf.write_embedded_message(8, graph_buf)

        return buf^
