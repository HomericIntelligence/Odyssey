"""Self-contained synthetic reproducer for module-level import chain JIT crash.

NO external dependencies beyond Mojo stdlib. This file creates synthetic
"heavy" modules in the same directory to simulate the import chain pattern
that crashes in ProjectOdyssey (1000-1500 line modules imported transitively
at module level).

HOW TO REPRODUCE (requires files repro_heavy_A.mojo and repro_heavy_B.mojo
in the same directory — see generate_synthetic_chain.sh):

  $ mojo test repro_import_chain_synthetic.mojo   # Expected: crash
  $ mojo test repro_import_chain_synthetic_fixed.mojo  # Expected: PASS

IMPORT CHAIN STRUCTURE:
  repro_import_chain_synthetic.mojo  (this file)
    → repro_heavy_B.mojo             (1400+ lines, module level import)
      → repro_heavy_A.mojo           (1400+ lines, module level import)

CRASH DIAGNOSIS:
  If the file crashes before printing "Running..." it is a module-level
  import explosion (compile time). If it prints "Running..." then crashes,
  the issue is different (runtime or accumulation).

ENVIRONMENT:
  Mojo 0.26.3 (dev2026040705), GLIBC 2.39, Linux 6.6.87 x86_64
"""

# Module-level imports — triggers compilation of the full chain at parse time
from repro_heavy_B import heavy_b_op_1, heavy_b_op_2


def test_module_level_import() raises:
    print("test_module_level_import: Reached test body")
    var result = heavy_b_op_1(42)
    var result2 = heavy_b_op_2(result)
    if result2 >= 0:
        print("test_module_level_import: PASS")
    else:
        print("test_module_level_import: FAIL")


def main() raises:
    print("Running: repro_import_chain_synthetic (module-level imports)")
    test_module_level_import()
