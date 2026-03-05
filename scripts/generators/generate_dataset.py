#!/usr/bin/env python3
"""
Generate boilerplate code for dataset loaders.

Usage:
    python scripts/generators/generate_dataset.py \\
        --name CustomDataset \\
        --format "image,label" \\
        --output datasets/custom_dataset.mojo

    python scripts/generators/generate_dataset.py \\
        --name TextDataset \\
        --format "text,label" \\
        --source csv \\
        --output datasets/text_dataset.mojo

Template Placeholders:
    The generated Mojo dataset files contain TEMPLATE: markers indicating
    sections that must be implemented for the specific data source and format.
    These are intentional scaffolding in the generated output, not
    implementation gaps in this script.
"""

import argparse
from datetime import datetime
from pathlib import Path

from scripts.generators.templates import to_snake_case


DATASET_TEMPLATES = {
    "image": """
struct {{name}}(Dataset):
    \"\"\"{{name}} image dataset.

    Loads images and labels from disk.
    \"\"\"

    var images: List[ExTensor]
    var labels: List[Int]
    var data_dir: String
    var transform: Optional[Transform]

    fn __init__(
        out self,
        data_dir: String,
        train: Bool = True,
        transform: Optional[Transform] = None
    ):
        \"\"\"Load dataset from directory.

        Args:
            data_dir: Path to dataset directory
            train: Whether to load training or test split
            transform: Optional data transformations
        \"\"\"
        self.data_dir = data_dir
        self.transform = transform
        self.images = List[ExTensor]()
        self.labels = List[Int]()

        # Load the appropriate split
        var split = "train" if train else "test"
        self._load_data(split)

    fn _load_data(mut self, split: String):
        \"\"\"Load images and labels from disk.

        Args:
            split: Data split to load ("train" or "test")
        \"\"\"
        # TEMPLATE: Implement loading logic
        # Example structure:
        # data_dir/
        #   train/
        #     class_0/
        #       img_001.png
        #     class_1/
        #       img_001.png
        #   test/
        #     ...
        raise Error("Not implemented: _load_data")

    fn __len__(self) -> Int:
        \"\"\"Get dataset size.

        Returns:
            Number of samples in dataset
        \"\"\"
        return len(self.images)

    fn __getitem__(self, idx: Int) -> Tuple[ExTensor, Int]:
        \"\"\"Get item by index.

        Args:
            idx: Sample index

        Returns:
            (image, label) tuple
        \"\"\"
        var image = self.images[idx]

        # Apply transforms if present
        if self.transform:
            image = self.transform.value()(image)

        return (image, self.labels[idx])

    fn get_batch(self, indices: List[Int]) -> Tuple[ExTensor, ExTensor]:
        \"\"\"Get a batch of samples.

        Args:
            indices: List of sample indices

        Returns:
            (images_batch, labels_batch) tuple
        \"\"\"
        var batch_images = List[ExTensor]()
        var batch_labels = List[Int]()

        for idx in indices:
            var sample = self[idx[]]
            batch_images.append(sample[0])
            batch_labels.append(sample[1])

        # Stack into batch tensors
        # TEMPLATE: Implement proper stacking
        return (batch_images[0], ExTensor.from_list(batch_labels))
""",
    "text": """
struct {{name}}(Dataset):
    \"\"\"{{name}} text dataset.

    Loads text data and labels.
    \"\"\"

    var texts: List[String]
    var labels: List[Int]
    var vocab: Dict[String, Int]
    var max_length: Int

    fn __init__(
        out self,
        data_path: String,
        max_length: Int = 512,
        vocab: Optional[Dict[String, Int]] = None
    ):
        \"\"\"Load dataset from file.

        Args:
            data_path: Path to data file (CSV, JSON, etc.)
            max_length: Maximum sequence length
            vocab: Optional pre-built vocabulary
        \"\"\"
        self.max_length = max_length
        self.texts = List[String]()
        self.labels = List[Int]()

        if vocab:
            self.vocab = vocab.value()
        else:
            self.vocab = Dict[String, Int]()

        self._load_data(data_path)

        if not vocab:
            self._build_vocab()

    fn _load_data(mut self, data_path: String):
        \"\"\"Load text data from file.

        Args:
            data_path: Path to data file
        \"\"\"
        # TEMPLATE: Implement loading logic
        # Supports: CSV, JSON, plain text
        raise Error("Not implemented: _load_data")

    fn _build_vocab(mut self):
        \"\"\"Build vocabulary from loaded texts.\"\"\"
        # TEMPLATE: Implement vocabulary building
        # Add special tokens: <PAD>, <UNK>, <BOS>, <EOS>
        self.vocab["<PAD>"] = 0
        self.vocab["<UNK>"] = 1
        raise Error("Not implemented: _build_vocab")

    fn _tokenize(self, text: String) -> List[Int]:
        \"\"\"Convert text to token indices.

        Args:
            text: Input text string

        Returns:
            List of token indices
        \"\"\"
        # TEMPLATE: Implement tokenization
        raise Error("Not implemented: _tokenize")

    fn __len__(self) -> Int:
        \"\"\"Get dataset size.\"\"\"
        return len(self.texts)

    fn __getitem__(self, idx: Int) -> Tuple[ExTensor, Int]:
        \"\"\"Get item by index.

        Args:
            idx: Sample index

        Returns:
            (token_ids, label) tuple
        \"\"\"
        var tokens = self._tokenize(self.texts[idx])

        # Pad or truncate to max_length
        while len(tokens) < self.max_length:
            tokens.append(self.vocab["<PAD>"])
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        return (ExTensor.from_list(tokens), self.labels[idx])
""",
    "tabular": """
struct {{name}}(Dataset):
    \"\"\"{{name}} tabular dataset.

    Loads structured tabular data.
    \"\"\"

    var features: ExTensor
    var labels: ExTensor
    var feature_names: List[String]
    var num_samples: Int
    var num_features: Int

    fn __init__(out self, data_path: String, target_column: String):
        \"\"\"Load dataset from CSV file.

        Args:
            data_path: Path to CSV file
            target_column: Name of target/label column
        \"\"\"
        self.feature_names = List[String]()
        self._load_csv(data_path, target_column)

    fn _load_csv(mut self, data_path: String, target_column: String):
        \"\"\"Load data from CSV file.

        Args:
            data_path: Path to CSV file
            target_column: Name of target column
        \"\"\"
        # TEMPLATE: Implement CSV loading
        # 1. Read header for feature names
        # 2. Parse numeric values
        # 3. Separate features and labels
        raise Error("Not implemented: _load_csv")

    fn __len__(self) -> Int:
        \"\"\"Get dataset size.\"\"\"
        return self.num_samples

    fn __getitem__(self, idx: Int) -> Tuple[ExTensor, ExTensor]:
        \"\"\"Get item by index.

        Args:
            idx: Sample index

        Returns:
            (features, label) tuple
        \"\"\"
        return (self.features[idx], self.labels[idx])

    fn normalize(mut self, mean: Optional[ExTensor] = None, std: Optional[ExTensor] = None):
        \"\"\"Normalize features to zero mean and unit variance.

        Args:
            mean: Optional precomputed mean
            std: Optional precomputed std
        \"\"\"
        # TEMPLATE: Implement normalization
        raise Error("Not implemented: normalize")
""",
    "custom": """
struct {{name}}(Dataset):
    \"\"\"{{name}} custom dataset.

    A template for custom data loading.
    \"\"\"

    var data: List[ExTensor]
    var targets: List[ExTensor]

    fn __init__(out self, data_path: String):
        \"\"\"Initialize dataset.

        Args:
            data_path: Path to data
        \"\"\"
        self.data = List[ExTensor]()
        self.targets = List[ExTensor]()
        self._load_data(data_path)

    fn _load_data(mut self, data_path: String):
        \"\"\"Load data from source.

        Args:
            data_path: Path to data
        \"\"\"
        # TEMPLATE: Implement custom loading logic
        raise Error("Not implemented: _load_data")

    fn __len__(self) -> Int:
        \"\"\"Get dataset size.\"\"\"
        return len(self.data)

    fn __getitem__(self, idx: Int) -> Tuple[ExTensor, ExTensor]:
        \"\"\"Get item by index.

        Args:
            idx: Sample index

        Returns:
            (data, target) tuple
        \"\"\"
        return (self.data[idx], self.targets[idx])
""",
}


def determine_format(format_str: str) -> str:
    """Determine dataset format from specification.

    Args:
        format_str: Format specification like "image,label" or "text,label"

    Returns:
        Format type: "image", "text", "tabular", or "custom"
    """
    format_lower = format_str.lower()
    if "image" in format_lower or "img" in format_lower:
        return "image"
    elif "text" in format_lower or "sentence" in format_lower:
        return "text"
    elif "csv" in format_lower or "tabular" in format_lower or "feature" in format_lower:
        return "tabular"
    return "custom"


def generate_dataset_code(
    name: str,
    format_type: str = "image",
) -> str:
    """Generate complete dataset code.

    Args:
        name: Dataset name (PascalCase)
        format_type: Type of dataset format

    Returns:
        Generated Mojo code
    """
    snake_name = to_snake_case(name)

    # Get template
    template = DATASET_TEMPLATES.get(format_type, DATASET_TEMPLATES["custom"])

    # Substitute variables
    code = template.replace("{{name}}", name)

    # Generate header
    header = f'''# {snake_name}.mojo
"""
{name} dataset loader.

Auto-generated by scripts/generators/generate_dataset.py
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

from shared.datasets import Dataset
from shared.core import ExTensor
'''

    return header + code


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate boilerplate code for datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate image dataset
    python scripts/generators/generate_dataset.py \\
        --name CustomImageDataset \\
        --format "image,label" \\
        --output datasets/custom_images.mojo

    # Generate text dataset
    python scripts/generators/generate_dataset.py \\
        --name SentimentDataset \\
        --format "text,label" \\
        --output datasets/sentiment.mojo

    # Generate tabular dataset
    python scripts/generators/generate_dataset.py \\
        --name HousingDataset \\
        --format "tabular" \\
        --output datasets/housing.mojo
        """,
    )

    parser.add_argument("--name", required=True, help="Dataset name (PascalCase)")
    parser.add_argument(
        "--format",
        default="image,label",
        help="Data format: image,label | text,label | tabular | custom",
    )
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing")

    args = parser.parse_args()

    # Determine format type
    format_type = determine_format(args.format)

    # Generate code
    code = generate_dataset_code(
        name=args.name,
        format_type=format_type,
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
