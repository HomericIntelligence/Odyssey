"""Minimal self-contained reproducer for KGEN JIT buffer overflow.

This file is a self-contained reproducer for a KGEN JIT crash that
occurs when compiling a module that:
  1. Imports std.python (CPython interop)
  2. Defines a struct with a List[String] field
  3. Defines multiple overloaded __init__ constructors (6+)
  4. Uses Dict[String, <that struct>]

The crash manifests as:
    mojo: error: execution crashed with signal: Aborted
    Aborted (core dumped)
    ... libKGENCompilerRTShared.so ...

The crash happens at JIT *compilation time* -- before main() is entered
and before any output is printed. This means:
  - It is NOT a runtime error in user code
  - It is NOT a logic bug in the module
  - It IS a KGEN internal buffer overflow (__fortify_fail_abort)

## Environment

- Mojo version: 0.26.3
- Platform: Linux x86_64 (GitHub Actions ubuntu-latest runner, ~7 GB RAM)
- Reproduces: 100% of the time in CI; 0% locally on developer machines
  (CI resource constraints appear to be required to trigger the overflow)

## Crash trace (from CI logs)

    mojo: error: execution crashed with signal: Aborted
    Aborted (core dumped)
    /proc/self/fd/3:
    #0  libKGENCompilerRTShared.so+0x6d4ab(__fortify_fail_abort+0x1b)
    #1  libKGENCompilerRTShared.so+0x6d461(__fortify_fail+0x21)
    #2  libKGENCompilerRTShared.so+0x6d419 (...)
    #3  libKGENCompilerRTShared.so+0x15f5a (...)
    ...

## How to run

    pixi run mojo repro/kgen_jit_overflow_minimal.mojo

Expected (CI): crash before any output with Aborted/libKGENCompilerRTShared.so trace
Expected (local): may run successfully (resource constraints differ from CI)
"""

from std.python import Python, PythonObject


struct Value(Copyable, Movable):
    """A union-like value type with multiple overloaded constructors."""

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

    def __init__(out self, var value: List[String]):
        self.type_tag = "list"
        self.int_val = 0
        self.float_val = 0.0
        self.str_val = ""
        self.bool_val = False
        self.list_val = value^

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
    """Container backed by Dict[String, Value]."""

    var data: Dict[String, Value]

    def __init__(out self):
        self.data = Dict[String, Value]()

    def set(mut self, key: String, value: Int):
        self.data[key] = Value(value)

    def set(mut self, key: String, value: Float64):
        self.data[key] = Value(value)

    def set(mut self, key: String, value: String):
        self.data[key] = Value(value)

    def set(mut self, key: String, value: Bool):
        self.data[key] = Value(value)

    def get_int(self, key: String) raises -> Int:
        return self.data[key].int_val

    def get_string(self, key: String) raises -> String:
        return self.data[key].str_val

    def use_python_interop(self) raises:
        """Exercises the std.python import that is referenced at module level."""
        var builtins = Python.import_module("builtins")
        _ = builtins.str("test")


def main() raises:
    print("If you see this, the KGEN crash did NOT occur.")
    var c = Container()
    c.set("lr", Float64(0.001))
    c.set("name", "test")
    c.set("epochs", 10)
    print("Container works, lr =", c.get_string("name"))
