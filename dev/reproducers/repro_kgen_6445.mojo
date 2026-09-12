"""Validation reproducer for modular/modular#6445 (KGEN JIT buffer overflow).

Adapted from the issue's minimal reproducer to Mojo 1.0.0 syntax:
- `def __init__(out self, var value: List[String])` -> `value: List[String]` + `.copy()`
- everything else preserved: std.python import, struct with List[String] field,
  6 overloaded __init__ constructors, Dict[String, Value] usage.

Original issue: KGEN crashes with `__fortify_fail_abort` at JIT compilation time
(CI-only, resource-constrained). Reported on mojo 0.26.3; reporter saw 10/10 clean
on 1.0.0b2.dev2026052506 (possibly fixed by #6413).

Run:
    mojo repro_kgen_6445.mojo
"""
from std.python import Python, PythonObject


struct Value(Copyable, Movable):
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
    var data: Dict[String, Value]

    def __init__(out self):
        self.data = Dict[String, Value]()

    def set(mut self, key: String, value: String):
        self.data[key] = Value(value)

    def get_string(self, key: String) raises -> String:
        return self.data[key].str_val


def main() raises:
    print("If you see this, the KGEN crash did NOT occur.")
    var c = Container()
    c.set("name", "test")
    print("Result:", c.get_string("name"))
