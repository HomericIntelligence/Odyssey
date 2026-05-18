# ML Odyssey Roadmap

This document describes what is currently shipped and the near-term direction
based on open GitHub issues. No dates are promised; progress is tracked via
issues and PRs.

## Current State

**7 neural network architectures** fully implemented with layerwise unit tests
(run on every PR) and end-to-end integration tests (run weekly with real datasets):

| Architecture | Paper |
| --- | --- |
| LeNet-5 | LeCun et al., 1998 |
| AlexNet | Krizhevsky et al., 2012 |
| VGG-16 | Simonyan & Zisserman, 2014 |
| ResNet-18 | He et al., 2015 |
| MobileNetV1 | Howard et al., 2017 |
| GoogLeNet | Szegedy et al., 2014 |

**Shared library** (`src/projectodyssey/`) provides tensor ops, autograd, layers, optimizers,
schedulers, mixed-precision training, checkpointing, and evaluation — all in Mojo.

~198K lines of Mojo code; 298+ tests across CI tiers.

## Near-Term Direction

Active work tracked in open issues:

- **Training infrastructure** — improvements to the `Trainer` API and data
  pipeline (see issues #5183, #5184).
- **Paper implementations** — additional architectures from `papers/`; priority
  is driven by community interest and issue upvotes.
- **Quality and observability** — version automation (#5327), security/privacy
  documentation (#5332), and general codebase health from the triage swarm
  (#5354).

## How to Contribute

1. Browse [open issues](https://github.com/HomericIntelligence/ProjectOdyssey/issues)
   for work that interests you.
2. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the branch → PR → auto-merge
   workflow.
3. Check [`docs/getting-started/`](docs/getting-started/) for setup instructions.

All changes go through a pull request — the `main` branch is protected.
