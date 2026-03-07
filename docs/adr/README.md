# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) that document significant
architectural decisions made in the ML Odyssey project.

## What is an ADR?

An Architecture Decision Record captures a single architecture decision, including:

- **Context**: The situation and factors that led to the decision
- **Decision**: The specific choice that was made
- **Consequences**: The expected outcomes, both positive and negative
- **Alternatives Considered**: Other options that were evaluated

## ADR Index

| ADR | Title | Status | Date |
| --- | ----- | ------ | ---- |
| [ADR-001](ADR-001-language-selection-tooling.md) | Pragmatic Hybrid Language Approach for Tooling | Accepted | 2025-11-10 |
| [ADR-002](ADR-002-gradient-struct-return-types.md) | Gradient Struct Return Types (Tuple Return Workaround) | Accepted | 2025-11-20 |
| [ADR-003](ADR-003-memory-pool-architecture.md) | Memory Pool Architecture for Small Tensor Allocations | Accepted | 2025-12-28 |
| [ADR-004](ADR-004-testing-strategy.md) | Two-Tier Testing Strategy | Accepted | 2025-12-28 |
| [ADR-005](ADR-005-agent-system-architecture.md) | Agent System Architecture | Accepted | 2025-12-28 |
| [ADR-006](ADR-006-benchmarking-infrastructure.md) | Benchmarking Infrastructure | Accepted | 2025-12-28 |
| [ADR-007](ADR-007-training-infrastructure.md) | Training Infrastructure | Accepted | 2025-12-28 |
| [ADR-008](ADR-008-coverage-tool-blocker.md) | Defer Code Coverage Until Mojo Tooling Available | Accepted | 2025-12-10 |
| [ADR-009](ADR-009-heap-corruption-workaround.md) | Heap Corruption Workaround for Mojo Runtime Bug | Accepted | 2025-12-30 |
| [ADR-010](ADR-010-fp16-simd-mojo-limitation.md) | FP16 SIMD Limitation in Mojo v0.26.1 | Accepted | 2026-03-07 |

## ADR Status Lifecycle

ADRs move through the following statuses:

1. **Proposed**: Initial draft under discussion
2. **Accepted**: Decision approved and implemented
3. **Deprecated**: No longer applies (superseded or context changed)
4. **Superseded**: Replaced by a newer ADR (includes reference to replacement)

## When to Create an ADR

Create an ADR when making decisions that:

- Affect system architecture or structure
- Have long-term impact on the codebase
- Involve significant trade-offs
- May not be obvious to future contributors
- Represent a change from standard practices

## ADR Template

See [template.md](template.md) for the standard ADR format used in this project.

## Quick Links

- [ADR Template](template.md)
- [CLAUDE.md](/CLAUDE.md) - Project guidelines
- [Mojo Guidelines](/.claude/shared/mojo-guidelines.md) - Mojo-specific patterns

## References

- [Michael Nygard's ADR Article](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR GitHub Organization](https://adr.github.io/)
