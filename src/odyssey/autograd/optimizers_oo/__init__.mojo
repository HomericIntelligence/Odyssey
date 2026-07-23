"""OO Variable/GradientTape optimizer wrappers.

This package contains `Optimizer` trait implementations for **every**
optimizer exposed on the autograd path. Each struct is a thin delegator
that holds hyperparameters + per-parameter state buffers and delegates
the algorithm to the canonical functional step in
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

All 24 OO optimizers below are re-exported from `odyssey.autograd.__init__`
and surfaced in `src/odyssey/__init__.mojo` for top-level convenience.

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

# Adam family (Adan, AdOpt) and Sophia
from odyssey.autograd.optimizers_oo.adam_family import Adan, AdOpt, Sophia

# Sign-momentum + adaptive scaling
from odyssey.autograd.optimizers_oo.lion import Lion
from odyssey.autograd.optimizers_oo.lars import LARS
from odyssey.autograd.optimizers_oo.ftrl import FTRLProximal

# Muon family (matrix-shaped params, Newton-Schulz orthogonalization)
from odyssey.autograd.optimizers_oo.muon_family import (
    Muon,
    NorMuon,
    MGUPMuon,
    MuonHyperball,
    LionMuon,
)

# Shampoo family (two-sided preconditioners)
from odyssey.autograd.optimizers_oo.shampoo_family import (
    Shampoo,
    SOAP,
    KLShampoo,
    SPlus,
)

# Schedule-Free family (anytime iterate averaging)
from odyssey.autograd.optimizers_oo.schedule_free import (
    ScheduleFree,
    ScheduleFreePlus,
    SFNorMuon,
)

# Parameter-free
from odyssey.autograd.optimizers_oo.prodigy import Prodigy
