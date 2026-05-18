# Papers and Shared Library Integration Guide

This guide explains how papers/ and src/projectodyssey/ directories work together.

## Overview

- **papers/**: Individual paper implementations
- **src/projectodyssey/**: Reusable ML/AI components

## Integration Pattern

Papers import from shared:

```mojo
```mojo

from projectodyssey.core import Layer, Module
from projectodyssey.training import Optimizer
from projectodyssey.data import Dataset

```text

## Quick Start

See [quick-start-new-paper.md](quick-start-new-paper.md) for creating new papers.
