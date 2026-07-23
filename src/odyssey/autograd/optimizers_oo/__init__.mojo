"""OO Variable/GradientTape optimizer wrappers.

This package contains `Optimizer` trait implementations for the optimizers
that currently expose a Variable / GradientTape API. Each struct is a thin
delegator that holds hyperparameters + per-parameter state buffers and
delegates the algorithm to the canonical functional step in
`src/odyssey/training/optimizers/<name>.mojo`.

State is held in `Dict[Int, AnyTensor]` keyed by `Variable.id` so the same
struct can drive a Variable list of arbitrary size and survive parameter
reordering.

Pattern:
    var tape = GradientTape()
    tape.enable()
    var params: List[Variable] = [...register with tape...]
    loss.backward(tape)
    optimizer.step(params, tape)
    optimizer.zero_grad(tape)

The 8 OO wrappers below are re-exported from `odyssey.autograd.__init__`
and surfaced in `src/odyssey/__init__.mojo` for top-level convenience.
The remaining optimizers in `src/odyssey/training/optimizers/` are
functional-only for now (their math is consumed via the canonical
`<name>_step` / `init_<name>_state` API; an OO wrapper at
`optimizers_oo/<name>.mojo` is the documented extension point per the
recipe in `docs/dev/optimizer-extension-guide.md`).

Coverage today (8 OO wrappers):
  - Core 5 (SGD, Adam, AdamW, AdaGrad, RMSprop) — moved here from the old
    inline classes in `src/odyssey/autograd/optimizers.mojo`.
  - Sibling singles (Lion, LARS, FTRLProximal) — promoted as single-file
    wrappers alongside the core 5.

Coverage queued for follow-up PRs (functional-only today):
  - Adan / AdOpt / Sophia (adam_family)
  - Muon / NorMuon / MGUPMuon / MuonHyperball / LionMuon (muon_family)
  - Shampoo / SOAP / KLShampoo / SPlus (shampoo_family)
  - ScheduleFree / ScheduleFreePlus / SFNorMuon (schedule_free family)
  - Prodigy

See `docs/dev/optimizer-extension-guide.md` for the contributor contract
that makes this package the **canonical extension point** for adding new
optimizers on the Variable/GradientTape API.
"""

# Core 5 — historically lived in `src/odyssey/autograd/optimizers.mojo`,
# now moved here as individual files to match the per-struct convention.
from odyssey.autograd.optimizers_oo.sgd import SGD
from odyssey.autograd.optimizers_oo.adam import Adam
from odyssey.autograd.optimizers_oo.adamw import AdamW
from odyssey.autograd.optimizers_oo.adagrad import AdaGrad
from odyssey.autograd.optimizers_oo.rmsprop import RMSprop

# Sibling singles (sign-momentum + adaptive scaling) — promoted as
# single-file wrappers alongside the core 5.
from odyssey.autograd.optimizers_oo.lion import Lion
from odyssey.autograd.optimizers_oo.lars import LARS
from odyssey.autograd.optimizers_oo.ftrl import FTRLProximal
