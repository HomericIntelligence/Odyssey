# Papers and Shared Library Integration Guide

This guide explains how papers/ and src/odyssey/ directories work together.

## Overview

- **papers/**: Individual paper implementations
- **src/odyssey/**: Reusable ML/AI components

## Integration Pattern

Papers import from shared:

```mojo
```mojo

from odyssey.core import Layer, Module
from odyssey.training import Optimizer
from odyssey.data import Dataset

```text

## Quick Start

See [quick-start-new-paper.md](quick-start-new-paper.md) for creating new papers.
