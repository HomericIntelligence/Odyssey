"""DEPRECATED: re-export shim — kept for backward compatibility only.

This module re-exports the 5 legacy OO optimizer structs so existing
callers continue to resolve the historical import path

    from odyssey.autograd.optimizers import SGD, Adam, AdamW, AdaGrad, RMSprop

unchanged. New code MUST import directly from
`odyssey.autograd.optimizers_oo`.

Canonical home for each struct (file + module path):

    SGD        ->  src/odyssey/autograd/optimizers_oo/sgd.mojo
                    (odyssey.autograd.optimizers_oo.sgd)
    Adam       ->  src/odyssey/autograd/optimizers_oo/adam.mojo
                    (odyssey.autograd.optimizers_oo.adam)
    AdamW      ->  src/odyssey/autograd/optimizers_oo/adamw.mojo
                    (odyssey.autograd.optimizers_oo.adamw)
    AdaGrad    ->  src/odyssey/autograd/optimizers_oo/adagrad.mojo
                    (odyssey.autograd.optimizers_oo.adagrad)
    RMSprop    ->  src/odyssey/autograd/optimizers_oo/rmsprop.mojo
                    (odyssey.autograd.optimizers_oo.rmsprop)

Each canonical struct is a thin delegator to the functional core in
`src/odyssey/training/optimizers/<name>.mojo`.

DO NOT add new struct bodies to this file. New optimizers belong in
`src/odyssey/autograd/optimizers_oo/<new_name>.mojo` and should only be
re-exported from this shim when a downstream caller still relies on the
historical path.

See `docs/dev/optimizer-extension-guide.md` for the canonical contributor
contract.
"""

# Re-export the 5 core OO optimizers from the canonical location so the
# historical path `from odyssey.autograd.optimizers import SGD, ...`
# keeps working.
from odyssey.autograd.optimizers_oo.sgd import SGD
from odyssey.autograd.optimizers_oo.adam import Adam
from odyssey.autograd.optimizers_oo.adamw import AdamW
from odyssey.autograd.optimizers_oo.adagrad import AdaGrad
from odyssey.autograd.optimizers_oo.rmsprop import RMSprop
