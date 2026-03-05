#!/usr/bin/env python3
"""
Generate boilerplate code for new models.

Usage:
    python scripts/generators/generate_model.py \\
        --name ResNet18 \\
        --type classification \\
        --output models/resnet18.mojo

    python scripts/generators/generate_model.py \\
        --name VAE \\
        --type generative \\
        --layers "encoder:Conv2d,decoder:ConvTranspose2d" \\
        --output models/vae.mojo

Template Placeholders:
    The generated Mojo model files contain TEMPLATE: markers indicating sections
    that must be implemented for the specific model architecture. These are
    intentional scaffolding in the generated output, not implementation gaps
    in this script. The Python fallback code (used when no layers are provided)
    also emits TEMPLATE: markers to signal that layers must be added.
"""

import argparse
from datetime import datetime
from pathlib import Path

from scripts.generators.templates import (
    generate_imports,
    to_snake_case,
)


MODEL_TEMPLATES = {
    "classification": """
struct {{name}}(Module):
    \"\"\"{{name}} classification model.

    A neural network for image classification tasks.
    \"\"\"

{{layer_declarations}}

    fn __init__(out self, num_classes: Int = {{num_classes}}):
        \"\"\"Initialize {{name}}.

        Args:
            num_classes: Number of output classes (default: {{num_classes}})
        \"\"\"
{{layer_initializations}}

    fn forward(self, input: ExTensor) -> ExTensor:
        \"\"\"Forward pass through the network.

        Args:
            input: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output logits of shape (batch, num_classes)
        \"\"\"
        var x = input

{{forward_pass}}

        return x

    fn parameters(self) -> List[ExTensor]:
        \"\"\"Get all trainable parameters.

        Returns:
            List of parameter tensors
        \"\"\"
        var params = List[ExTensor]()
{{parameter_collection}}
        return params
""",
    "generative": """
struct {{name}}(Module):
    \"\"\"{{name}} generative model.

    A neural network for generating data.
    \"\"\"

{{layer_declarations}}
    var latent_dim: Int

    fn __init__(out self, latent_dim: Int = {{latent_dim}}):
        \"\"\"Initialize {{name}}.

        Args:
            latent_dim: Dimension of latent space (default: {{latent_dim}})
        \"\"\"
        self.latent_dim = latent_dim
{{layer_initializations}}

    fn encode(self, input: ExTensor) -> ExTensor:
        \"\"\"Encode input to latent space.

        Args:
            input: Input tensor

        Returns:
            Latent representation
        \"\"\"
        var x = input
        # TEMPLATE: Implement encoder
        return x

    fn decode(self, z: ExTensor) -> ExTensor:
        \"\"\"Decode latent representation.

        Args:
            z: Latent tensor

        Returns:
            Reconstructed output
        \"\"\"
        var x = z
        # TEMPLATE: Implement decoder
        return x

    fn forward(self, input: ExTensor) -> ExTensor:
        \"\"\"Forward pass (encode then decode).

        Args:
            input: Input tensor

        Returns:
            Reconstructed output
        \"\"\"
        var z = self.encode(input)
        return self.decode(z)

    fn parameters(self) -> List[ExTensor]:
        \"\"\"Get all trainable parameters.\"\"\"
        var params = List[ExTensor]()
{{parameter_collection}}
        return params
""",
    "detection": """
struct {{name}}(Module):
    \"\"\"{{name}} object detection model.

    A neural network for detecting objects in images.
    \"\"\"

{{layer_declarations}}
    var num_classes: Int
    var num_anchors: Int

    fn __init__(out self, num_classes: Int = {{num_classes}}, num_anchors: Int = 9):
        \"\"\"Initialize {{name}}.

        Args:
            num_classes: Number of object classes
            num_anchors: Number of anchor boxes per location
        \"\"\"
        self.num_classes = num_classes
        self.num_anchors = num_anchors
{{layer_initializations}}

    fn forward(self, input: ExTensor) -> Tuple[ExTensor, ExTensor]:
        \"\"\"Forward pass returning class scores and bounding boxes.

        Args:
            input: Input image tensor of shape (batch, 3, H, W)

        Returns:
            Tuple of (class_scores, bbox_predictions)
        \"\"\"
        var x = input

{{forward_pass}}

        # TEMPLATE: Return class scores and bbox predictions
        return (x, x)

    fn parameters(self) -> List[ExTensor]:
        \"\"\"Get all trainable parameters.\"\"\"
        var params = List[ExTensor]()
{{parameter_collection}}
        return params
""",
    "segmentation": """
struct {{name}}(Module):
    \"\"\"{{name}} segmentation model.

    A neural network for semantic segmentation.
    \"\"\"

{{layer_declarations}}
    var num_classes: Int

    fn __init__(out self, num_classes: Int = {{num_classes}}):
        \"\"\"Initialize {{name}}.

        Args:
            num_classes: Number of segmentation classes
        \"\"\"
        self.num_classes = num_classes
{{layer_initializations}}

    fn forward(self, input: ExTensor) -> ExTensor:
        \"\"\"Forward pass returning pixel-wise predictions.

        Args:
            input: Input image tensor of shape (batch, 3, H, W)

        Returns:
            Segmentation mask of shape (batch, num_classes, H, W)
        \"\"\"
        var x = input

{{forward_pass}}

        return x

    fn parameters(self) -> List[ExTensor]:
        \"\"\"Get all trainable parameters.\"\"\"
        var params = List[ExTensor]()
{{parameter_collection}}
        return params
""",
}


def parse_layers(layers_str: str) -> list[tuple[str, str, dict]]:
    """Parse layer specification string.

    Args:
        layers_str: Comma-separated layer specs like "conv1:Conv2d(64),bn1:BatchNorm2d"

    Returns:
        List of (name, type, params) tuples
    """
    if not layers_str:
        return []

    layers = []
    for spec in layers_str.split(","):
        spec = spec.strip()
        if ":" not in spec:
            continue

        name, type_spec = spec.split(":", 1)
        name = name.strip()
        type_spec = type_spec.strip()

        # Parse parameters if present
        params = {}
        if "(" in type_spec:
            type_name = type_spec[: type_spec.index("(")]
            param_str = type_spec[type_spec.index("(") + 1 : type_spec.rindex(")")]
            for param in param_str.split(","):
                param = param.strip()
                if "=" in param:
                    k, v = param.split("=", 1)
                    params[k.strip()] = v.strip()
                elif param.isdigit():
                    params["out_features"] = param
        else:
            type_name = type_spec

        layers.append((name, type_name, params))

    return layers


def generate_layer_declarations(layers: list[tuple[str, str, dict]]) -> str:
    """Generate layer variable declarations."""
    if not layers:
        return "    # TEMPLATE: Add layer declarations\n    var placeholder: Linear"

    lines = []
    for name, layer_type, _ in layers:
        lines.append(f"    var {name}: {layer_type}")
    return "\n".join(lines)


def generate_layer_initializations(layers: list[tuple[str, str, dict]], indent: int = 8) -> str:
    """Generate layer initialization code."""
    if not layers:
        return " " * indent + "self.placeholder = Linear(1, 1)"

    lines = []
    prefix = " " * indent
    for name, layer_type, params in layers:
        if params:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            lines.append(f"{prefix}self.{name} = {layer_type}({param_str})")
        else:
            lines.append(f"{prefix}self.{name} = {layer_type}()")
    return "\n".join(lines)


def generate_forward_pass(layers: list[tuple[str, str, dict]], indent: int = 8) -> str:
    """Generate forward pass code."""
    if not layers:
        return " " * indent + "# TEMPLATE: Implement forward pass\n" + " " * indent + "x = self.placeholder(x)"

    lines = []
    prefix = " " * indent
    for name, layer_type, _ in layers:
        lines.append(f"{prefix}x = self.{name}(x)")
    return "\n".join(lines)


def generate_parameter_collection(layers: list[tuple[str, str, dict]], indent: int = 8) -> str:
    """Generate parameter collection code."""
    if not layers:
        return " " * indent + "params.extend(self.placeholder.parameters())"

    lines = []
    prefix = " " * indent
    for name, _, _ in layers:
        lines.append(f"{prefix}params.extend(self.{name}.parameters())")
    return "\n".join(lines)


def generate_model_code(
    name: str,
    model_type: str = "classification",
    layers: list[tuple[str, str, dict]] | None = None,
    num_classes: int = 10,
    latent_dim: int = 128,
) -> str:
    """Generate complete model code.

    Args:
        name: Model name (PascalCase)
        model_type: Type of model (classification, generative, detection, segmentation)
        layers: List of layer specifications
        num_classes: Number of output classes
        latent_dim: Latent dimension for generative models

    Returns:
        Generated Mojo code
    """
    layers = layers or []

    # Get layer types for imports
    layer_types = [layer_type for _, layer_type, _ in layers]

    # Generate code sections
    layer_decls = generate_layer_declarations(layers)
    layer_inits = generate_layer_initializations(layers)
    forward = generate_forward_pass(layers)
    params = generate_parameter_collection(layers)

    # Get template
    template = MODEL_TEMPLATES.get(model_type, MODEL_TEMPLATES["classification"])

    # Substitute variables
    code = template.replace("{{name}}", name)
    code = code.replace("{{num_classes}}", str(num_classes))
    code = code.replace("{{latent_dim}}", str(latent_dim))
    code = code.replace("{{layer_declarations}}", layer_decls)
    code = code.replace("{{layer_initializations}}", layer_inits)
    code = code.replace("{{forward_pass}}", forward)
    code = code.replace("{{parameter_collection}}", params)

    # Generate header
    snake_name = to_snake_case(name)
    imports = generate_imports(layer_types)

    header = f'''# {snake_name}.mojo
"""
{name} model for {model_type}.

Auto-generated by scripts/generators/generate_model.py
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

{imports}
'''

    return header + code


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate boilerplate code for new models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate classification model
    python scripts/generators/generate_model.py \\
        --name ResNet18 \\
        --type classification \\
        --num-classes 1000 \\
        --output models/resnet18.mojo

    # Generate with layer specs
    python scripts/generators/generate_model.py \\
        --name SimpleCNN \\
        --type classification \\
        --layers "conv1:Conv2d(3,64),bn1:BatchNorm2d(64),relu:ReLU,fc:Linear(64,10)" \\
        --output models/simple_cnn.mojo
        """,
    )

    parser.add_argument("--name", required=True, help="Model name (PascalCase)")
    parser.add_argument(
        "--type",
        choices=["classification", "generative", "detection", "segmentation"],
        default="classification",
        help="Model type (default: classification)",
    )
    parser.add_argument(
        "--layers",
        help="Layer specifications: name:Type(params),name2:Type2",
    )
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing")

    args = parser.parse_args()

    # Parse layers
    layers = parse_layers(args.layers) if args.layers else []

    # Generate code
    code = generate_model_code(
        name=args.name,
        model_type=args.type,
        layers=layers,
        num_classes=args.num_classes,
        latent_dim=args.latent_dim,
    )

    # Write output
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        print(f"Error: {output_path} already exists. Use --force to overwrite.")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(code)
    print(f"Generated: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
