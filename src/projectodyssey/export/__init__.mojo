# src/projectodyssey/export/__init__.mojo
"""
Model export utilities for ML Odyssey.

Provides export functionality for deploying models to production:
- ONNX export for cross-framework interoperability
- Model tracing for operation capture

Usage:
    from projectodyssey.export import ONNXExporter, trace_model

    var exporter = ONNXExporter()
    var graph = trace_model(model, sample_input)
    exporter.export_graph(graph, "model.onnx")
"""
