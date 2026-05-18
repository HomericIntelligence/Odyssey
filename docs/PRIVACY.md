# Privacy & Data-Handling Policy

ML Odyssey is a **local Mojo-based ML training framework**, not a hosted service.
This document explains what data the framework can process, who is responsible for
privacy compliance, and practical recommendations for operators.

## What Data the Framework Can Process

| Category | Examples | Storage |
| --- | --- | --- |
| Training datasets | EMNIST, CIFAR-10, user-provided files | Local disk only |
| Model weights | Checkpoint files written by `src/projectodyssey/training/checkpoint.mojo` | Local disk only |
| Training metrics | Loss, accuracy tracked in memory by `src/projectodyssey/training/metrics/` | In-process only |

All data stays on the machine running training. No data is transmitted externally.

## Telemetry and Phone-Home

ProjectOdyssey contains **no telemetry, no analytics, and no network calls** of
any kind. The codebase (`src/projectodyssey/`, `examples/`) has been audited and contains no
outbound HTTP, socket, or reporting code. Training runs are entirely local.

Note: the Mojo compiler and Modular toolchain are separate products with their own
privacy policy: <https://www.modular.com/legal/privacy>.

## Operator Responsibility

ML Odyssey is a **library, not a service**. Privacy and data-protection obligations
under GDPR, CCPA, or equivalent regulations fall on the **operator** — the person or
organisation that configures and runs training.

Operators are responsible for:

- Ensuring they have a lawful basis to process any personal data in training datasets
- Applying appropriate access controls to dataset files and checkpoint directories
- Complying with data-subject rights (access, erasure, portability) for any PII in datasets

## Recommendations for Handling PII in Training Data

If your training dataset may contain personal data, apply these practices before
passing data to the framework:

1. **Pseudonymise or anonymise** — replace direct identifiers (names, IDs, email
   addresses) with pseudonyms or synthetic values before loading data.
2. **Dataset cards** — document what your dataset contains, its source, and what
   preprocessing has been applied (e.g., a `DATASET_CARD.md` alongside your data files).
3. **Retention limits** — delete raw datasets and intermediate checkpoints when they
   are no longer needed for training or reproducibility.
4. **Access controls** — restrict filesystem permissions on dataset directories
   (`chmod 700`) so only the training process user can read them.

## Further Reading

- [SECURITY.md][security-link] — vulnerability reporting and security best practices
- [Modular AI Privacy Policy](https://www.modular.com/legal/privacy) — privacy policy
  for the Mojo compiler and Modular toolchain

[security-link]: https://github.com/HomericIntelligence/ProjectOdyssey/blob/main/SECURITY.md
