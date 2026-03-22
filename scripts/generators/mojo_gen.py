#!/usr/bin/env python3
"""
Unified code generation CLI for ML Odyssey.

A command-line tool for generating Mojo boilerplate code including
models, layers, datasets, training scripts, and tests.

Usage:
    python -m scripts.generators.mojo_gen model ResNet18 --type classification -o models/
    python -m scripts.generators.mojo_gen layer Attention --inputs "q,k,v" -o shared/nn/
    python -m scripts.generators.mojo_gen dataset ImageFolder --format directory -o datasets/
    python -m scripts.generators.mojo_gen training --model MyModel --dataset MyDataset -o examples/
    python -m scripts.generators.mojo_gen tests --module shared.nn.linear --type layer -o tests/

Or using the convenience script:
    ./scripts/mojo-gen model ResNet18 --type classification
"""

import argparse
import sys
from pathlib import Path


def cmd_model(args):
    """Generate model code."""
    from scripts.generators.generate_model import generate_model_code, parse_layers

    layers = parse_layers(args.layers) if args.layers else []

    code = generate_model_code(
        name=args.name,
        model_type=args.type,
        layers=layers,
        num_classes=args.num_classes,
        latent_dim=args.latent_dim,
    )

    output_path = Path(args.output)
    if output_path.is_dir():
        from scripts.generators.templates import to_snake_case

        output_path = output_path / f"{to_snake_case(args.name)}.mojo"

    if output_path.exists() and not args.force:
        print(f"Error: {output_path} already exists. Use --force to overwrite.")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(code)
    print(f"Generated model: {output_path}")
    return 0


def cmd_layer(args):
    """Generate layer code."""
    from scripts.generators.generate_layer import (
        generate_layer_code,
        parse_inputs,
        parse_params,
    )

    inputs = parse_inputs(args.inputs)
    params = parse_params(args.params) if args.params else []

    code = generate_layer_code(
        name=args.name,
        inputs=inputs,
        params=params,
        has_parameters=not args.no_parameters,
    )

    output_path = Path(args.output)
    if output_path.is_dir():
        from scripts.generators.templates import to_snake_case

        output_path = output_path / f"{to_snake_case(args.name)}.mojo"

    if output_path.exists() and not args.force:
        print(f"Error: {output_path} already exists. Use --force to overwrite.")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(code)
    print(f"Generated layer: {output_path}")
    return 0


def cmd_dataset(args):
    """Generate dataset code."""
    from scripts.generators.generate_dataset import generate_dataset_code, determine_format

    format_type = determine_format(args.format)

    code = generate_dataset_code(
        name=args.name,
        format_type=format_type,
    )

    output_path = Path(args.output)
    if output_path.is_dir():
        from scripts.generators.templates import to_snake_case

        output_path = output_path / f"{to_snake_case(args.name)}.mojo"

    if output_path.exists() and not args.force:
        print(f"Error: {output_path} already exists. Use --force to overwrite.")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(code)
    print(f"Generated dataset: {output_path}")
    return 0


def cmd_training(args):
    """Generate training script."""
    from scripts.generators.generate_training_script import generate_training_code
    from scripts.generators.templates import to_snake_case

    code = generate_training_code(
        model=args.model,
        dataset=args.dataset,
        optimizer=args.optimizer,
        loss=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    output_path = Path(args.output)
    if output_path.is_dir():
        output_path = output_path / f"train_{to_snake_case(args.model)}.mojo"

    if output_path.exists() and not args.force:
        print(f"Error: {output_path} already exists. Use --force to overwrite.")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(code)
    print(f"Generated training script: {output_path}")
    return 0


def cmd_tests(args):
    """Generate test code."""
    from scripts.generators.generate_tests import generate_test_code

    code = generate_test_code(
        module_path=args.module,
        test_type=args.type,
    )

    output_path = Path(args.output)
    if output_path.is_dir():
        module_name = args.module.split(".")[-1]
        output_path = output_path / f"test_{module_name}.mojo"

    if output_path.exists() and not args.force:
        print(f"Error: {output_path} already exists. Use --force to overwrite.")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(code)
    print(f"Generated tests: {output_path}")
    return 0


def cmd_list(args):
    """List available generators and templates."""
    print("ML Odyssey Code Generators")
    print("=" * 50)
    print()
    print("Available generators:")
    print()
    print("  model      Generate model boilerplate")
    print("             Types: classification, generative, detection, segmentation")
    print()
    print("  layer      Generate layer boilerplate")
    print("             Supports custom inputs, parameters, trainable/non-trainable")
    print()
    print("  dataset    Generate dataset loader boilerplate")
    print("             Formats: image, text, tabular, custom")
    print()
    print("  training   Generate complete training script")
    print("             Optimizers: sgd, adam, adamw, lars, lamb")
    print("             Losses: crossentropy, mse, bce, nll")
    print()
    print("  tests      Generate test boilerplate")
    print("             Types: unit, layer, model")
    print()
    print("Examples:")
    print()
    print("  python -m scripts.generators.mojo_gen model ResNet18 --type classification -o models/")
    print("  python -m scripts.generators.mojo_gen layer Attention --inputs 'q,k,v' -o shared/nn/")
    print("  python -m scripts.generators.mojo_gen dataset CIFAR10 --format image -o datasets/")
    print("  python -m scripts.generators.mojo_gen training --model LeNet5 --dataset MNIST -o examples/")
    print("  python -m scripts.generators.mojo_gen tests --module shared.nn.linear --type layer -o tests/")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mojo-gen",
        description="ML Odyssey Code Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Use 'mojo-gen <command> --help' for command-specific help.
Use 'mojo-gen list' to see all available generators.
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Generator command")

    # List command
    list_parser = subparsers.add_parser("list", help="List available generators")
    list_parser.set_defaults(func=cmd_list)

    # Model command
    model_parser = subparsers.add_parser("model", help="Generate model code")
    model_parser.add_argument("name", help="Model name (PascalCase)")
    model_parser.add_argument(
        "--type",
        "-t",
        choices=["classification", "generative", "detection", "segmentation"],
        default="classification",
        help="Model type",
    )
    model_parser.add_argument("--layers", "-l", help="Layer specs: name:Type,name2:Type2")
    model_parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    model_parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    model_parser.add_argument("--output", "-o", required=True, help="Output path")
    model_parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing")
    model_parser.set_defaults(func=cmd_model)

    # Layer command
    layer_parser = subparsers.add_parser("layer", help="Generate layer code")
    layer_parser.add_argument("name", help="Layer name (PascalCase)")
    layer_parser.add_argument("--inputs", "-i", default="input:AnyTensor", help="Input specs")
    layer_parser.add_argument("--params", "-p", help="Parameter specs")
    layer_parser.add_argument("--no-parameters", action="store_true", help="No trainable params")
    layer_parser.add_argument("--output", "-o", required=True, help="Output path")
    layer_parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing")
    layer_parser.set_defaults(func=cmd_layer)

    # Dataset command
    dataset_parser = subparsers.add_parser("dataset", help="Generate dataset code")
    dataset_parser.add_argument("name", help="Dataset name (PascalCase)")
    dataset_parser.add_argument("--format", "-m", default="image", help="Data format")
    dataset_parser.add_argument("--output", "-o", required=True, help="Output path")
    dataset_parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing")
    dataset_parser.set_defaults(func=cmd_dataset)

    # Training command
    training_parser = subparsers.add_parser("training", help="Generate training script")
    training_parser.add_argument("--model", "-m", required=True, help="Model name")
    training_parser.add_argument("--dataset", "-d", required=True, help="Dataset name")
    training_parser.add_argument("--optimizer", default="adam", help="Optimizer type")
    training_parser.add_argument("--loss", default="crossentropy", help="Loss function")
    training_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    training_parser.add_argument("--batch-size", type=int, help="Batch size")
    training_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    training_parser.add_argument("--weight-decay", type=float, help="Weight decay")
    training_parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    training_parser.add_argument("--output", "-o", required=True, help="Output path")
    training_parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing")
    training_parser.set_defaults(func=cmd_training)

    # Tests command
    tests_parser = subparsers.add_parser("tests", help="Generate test code")
    tests_parser.add_argument("--module", "-m", required=True, help="Module path to test")
    tests_parser.add_argument(
        "--type",
        "-t",
        choices=["unit", "layer", "model"],
        default="unit",
        help="Test type",
    )
    tests_parser.add_argument("--output", "-o", required=True, help="Output path")
    tests_parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing")
    tests_parser.set_defaults(func=cmd_tests)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
