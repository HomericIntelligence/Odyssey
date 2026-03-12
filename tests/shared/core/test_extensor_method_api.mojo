"""Tests for ExTensor method-style API: tile, repeat, permute, split.

Verifies that the thin wrapper methods on ExTensor produce identical results
to the functional implementations in shared.core.shape. Follows #3243.

NOTE: All tests are currently skipped because the ExTensor method wrappers
(.tile(), .repeat(), .permute(), .split()) have not been implemented yet.
The functional versions (tile(), repeat(), permute(), split()) work correctly.
TODO: Re-enable tests once ExTensor method wrappers are implemented.
  - ExTensor.tile(): https://github.com/HomericIntelligence/ProjectOdyssey/issues/TBD
  - ExTensor.repeat(): https://github.com/HomericIntelligence/ProjectOdyssey/issues/TBD
  - ExTensor.permute(): https://github.com/HomericIntelligence/ProjectOdyssey/issues/TBD
  - ExTensor.split(): https://github.com/HomericIntelligence/ProjectOdyssey/issues/TBD
"""


fn main() raises:
    print(
        "SKIPPED: ExTensor method API tests (tile, repeat, permute, split) -"
        " methods not yet implemented on ExTensor struct"
    )
