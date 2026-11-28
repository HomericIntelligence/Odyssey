"""Minimal reproducer for Mojo's virtual-address-space reservation.

The simplest possible Mojo program. Under a tight `ulimit -v` (virtual
memory cap), the compiler/runtime cannot reserve its address space and
the process aborts before printing anything. The official documented
minimum for Mojo development is 8 GiB RAM (mojolang.org/docs/requirements/);
this reproducer measures the actual virtual-limit threshold on the
current toolchain.

Run (measuring the crash threshold):
    for limit_kb in 2097152 2621440 3145728 3670016 4194304; do
        echo -n "${limit_kb} KB: "
        (ulimit -v $limit_kb; mojo run repro_vm_limit_hello.mojo >/dev/null 2>&1 && echo PASS || echo CRASH)
    done
"""


def main():
    print("hello")
