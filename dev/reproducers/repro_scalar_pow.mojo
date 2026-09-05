"""BUG REPRODUCER (Mojo 1.0.0 regression): `Scalar[dt] ** 0.5` fails to compile.

Environment: passes on 1.0.0b2 (2cf4d08a), fails on 1.0.0 stable (ed45d567).

Raising Scalar[dt] to 0.5 fails instantiation for float16 / bfloat16 / float32
with "constraint failed: unsupported type combination" (pass-manager error).
float64 works on both.

Run:
    mojo repro_scalar_pow.mojo
"""


def f[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    return x**0.5


def main():
    var v1 = Scalar[DType.float16](4.0)
    print("float16 =>", f(v1))
    var v2 = Scalar[DType.float32](4.0)
    print("float32 =>", f(v2))
    var v3 = Scalar[DType.float64](4.0)
    print("float64 =>", f(v3))
