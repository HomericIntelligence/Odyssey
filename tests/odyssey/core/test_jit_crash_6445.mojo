"""Permanent JIT canary for modular/modular#6445.

The historical failure was a KGEN JIT compile-time crash (`__fortify_fail_abort`
in libKGENCompilerRTShared.so, before any user code ran) on mojo 0.26.3,
triggered by the combination of: a module-level `std.python` import, a struct
with a `List[String]` field, 6 overloaded `__init__` constructors, and use of
that struct as a `Dict[String, Value]` value type. It was CI-only (resource
constrained) and is believed fixed by the #6413 JIT-stability fix
(`1.0.0b2.dev2026052506`); validation on 1.0.0b2 and 1.0.0 stable shows 10/10
clean plus a clean `tests/configs/` group (see docs/dev/mojo-1.0.0-regressions.md).

Keep this test small, keep all four trigger factors, and run it without any
CPU-target override. On a vulnerable toolchain the compile aborts with
`__fortify_fail_abort` before this file's first print; on a fixed toolchain it
completes normally.
"""

from std.python import Python, PythonObject


struct Value(Copyable, Movable):
    """Config-value struct: List[String] field + 6 overloaded constructors."""

    var type_tag: String
    var int_val: Int
    var float_val: Float64
    var str_val: String
    var bool_val: Bool
    var list_val: List[String]

    def __init__(out self, value: Int):
        self.type_tag = "int"
        self.int_val = value
        self.float_val = 0.0
        self.str_val = ""
        self.bool_val = False
        self.list_val = List[String]()

    def __init__(out self, value: Float64):
        self.type_tag = "float"
        self.int_val = 0
        self.float_val = value
        self.str_val = ""
        self.bool_val = False
        self.list_val = List[String]()

    def __init__(out self, value: String):
        self.type_tag = "string"
        self.int_val = 0
        self.float_val = 0.0
        self.str_val = value
        self.bool_val = False
        self.list_val = List[String]()

    def __init__(out self, value: Bool):
        self.type_tag = "bool"
        self.int_val = 0
        self.float_val = 0.0
        self.str_val = ""
        self.bool_val = value
        self.list_val = List[String]()

    def __init__(out self, value: List[String]):
        self.type_tag = "list"
        self.int_val = 0
        self.float_val = 0.0
        self.str_val = ""
        self.bool_val = False
        self.list_val = value.copy()

    def __init__(out self, value: List[Int]):
        self.type_tag = "list"
        self.int_val = 0
        self.float_val = 0.0
        self.str_val = ""
        self.bool_val = False
        self.list_val = List[String]()
        for i in range(len(value)):
            self.list_val.append(String(value[i]))


struct Container(Copyable, Movable):
    """Dict[String, Value] use — the fourth trigger factor."""

    var data: Dict[String, Value]

    def __init__(out self):
        self.data = Dict[String, Value]()

    def set(mut self, key: String, value: String):
        self.data[key] = Value(value)

    def get_string(self, key: String) raises -> String:
        return self.data[key].str_val


def test_value_overloads_and_dict() raises:
    """Exercise all constructors + Dict storage on the KGEN-sensitive path."""
    var c = Container()
    c.set("name", "test")
    c.data["int"] = Value(42)
    c.data["float"] = Value(1.5)
    c.data["bool"] = Value(True)
    var strings = List[String]()
    strings.append("a")
    strings.append("b")
    c.data["strings"] = Value(strings)
    var ints = List[Int]()
    ints.append(1)
    ints.append(2)
    ints.append(3)
    c.data["ints"] = Value(ints)
    if c.get_string("name") != "test":
        raise Error("expected get_string('name') == 'test'")


def test_python_interop_path() raises:
    """Exercise the Python interop JIT path (as in the #6413 canary)."""
    # Repeated imports keep this canary useful if the JIT regression is
    # intermittent while remaining cheap enough for every comprehensive run.
    for _ in range(200):
        var os_module = Python.import_module("os")
        _ = os_module


def main() raises:
    print("Running modular/modular#6445 JIT canary...")
    test_value_overloads_and_dict()
    test_python_interop_path()
    print("PASS: modular/modular#6445 JIT canary")
