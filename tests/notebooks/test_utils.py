"""Tests for notebook utility modules."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# matplotlib lives in the notebook feature env, not the default env.
# Skip visualization tests gracefully when it is unavailable.
matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed in default env")

# Add notebooks to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from notebooks.utils import tensor_utils, visualization


class TestTensorUtils:
    """Tests for tensor_utils module."""

    def test_numpy_to_mojo_binary_creates_file(self):
        """Test that numpy_to_mojo_binary creates a binary file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            output_path = Path(tmpdir) / "test.bin"

            tensor_utils.numpy_to_mojo_binary(array, str(output_path))

            assert output_path.exists(), "Binary file not created"
            assert output_path.stat().st_size > 0, "Binary file is empty"

    def test_mojo_binary_to_numpy_roundtrip(self):
        """Test that we can save and load tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = np.random.randn(4, 5).astype(np.float32)
            output_path = Path(tmpdir) / "tensor.bin"

            tensor_utils.numpy_to_mojo_binary(original, str(output_path))
            loaded = tensor_utils.mojo_binary_to_numpy(str(output_path), shape=(4, 5), dtype="float32")

            np.testing.assert_array_almost_equal(original, loaded)

    def test_mojo_binary_to_numpy_nonexistent_file(self):
        """Test that loading a nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            tensor_utils.mojo_binary_to_numpy("/nonexistent/path.bin", shape=(2, 2))

    def test_save_and_load_tensor_metadata(self):
        """Test tensor metadata saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            metadata_path = Path(tmpdir) / "metadata.json"

            tensor_utils.save_tensor_to_json(array, str(metadata_path))

            metadata = tensor_utils.load_tensor_metadata(str(metadata_path))

            assert metadata["shape"] == [2, 2]
            assert metadata["dtype_name"] == "float32"
            assert metadata["size"] == 4
            assert abs(metadata["min"] - 1.0) < 0.01
            assert abs(metadata["max"] - 4.0) < 0.01

    def test_compare_tensors_identical(self):
        """Test comparing identical tensors."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = tensor_utils.compare_tensors(a, b)

        assert result["equal"] is True
        assert result["max_difference"] == 0.0
        assert result["num_different_elements"] == 0

    def test_compare_tensors_with_differences(self):
        """Test comparing tensors with small differences."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[1.01, 2.01], [3.01, 4.01]])

        result = tensor_utils.compare_tensors(a, b, atol=0.001)

        assert result["equal"] is False
        assert result["max_difference"] > 0
        assert result["num_different_elements"] > 0

    def test_compare_tensors_shape_mismatch(self):
        """Test comparing tensors with different shapes."""
        a = np.array([[1.0, 2.0]])
        b = np.array([[1.0], [2.0]])

        result = tensor_utils.compare_tensors(a, b)

        assert result["equal"] is False
        assert "error" in result or "Shape mismatch" in str(result)


class TestVisualization:
    """Tests for visualization module."""

    def test_plot_training_curves_returns_figure(self):
        """Test that plot_training_curves returns a matplotlib figure."""
        import matplotlib.pyplot as plt

        losses = [1.0, 0.5, 0.2, 0.1]
        fig = visualization.plot_training_curves(losses)

        assert fig is not None
        assert hasattr(fig, "axes"), "Should return a matplotlib figure"
        plt.close(fig)

    def test_plot_training_curves_with_all_data(self):
        """Test plotting with loss, accuracy, and val data."""
        import matplotlib.pyplot as plt

        train_losses = [1.0, 0.5, 0.2]
        val_losses = [1.1, 0.6, 0.25]
        train_accs = [0.6, 0.8, 0.9]
        val_accs = [0.55, 0.75, 0.88]

        fig = visualization.plot_training_curves(train_losses, val_losses, train_accs, val_accs)

        assert fig is not None
        assert len(fig.axes) == 2, "Should have 2 subplots"
        plt.close(fig)

    def test_plot_confusion_matrix_returns_figure(self):
        """Test that plot_confusion_matrix returns a figure."""
        import matplotlib.pyplot as plt

        cm = np.array([[90, 10], [5, 95]])

        fig = visualization.plot_confusion_matrix(cm)

        assert fig is not None
        assert hasattr(fig, "axes"), "Should return a matplotlib figure"
        plt.close(fig)

    def test_plot_confusion_matrix_with_labels(self):
        """Test confusion matrix with class names."""
        import matplotlib.pyplot as plt

        cm = np.array([[90, 10], [5, 95]])
        class_names = ["Class A", "Class B"]

        fig = visualization.plot_confusion_matrix(cm, class_names)

        assert fig is not None
        plt.close(fig)

    def test_visualize_tensor_2d(self):
        """Test tensor visualization with 2D tensor."""
        import matplotlib.pyplot as plt

        tensor = np.random.randn(10, 10)
        fig = visualization.visualize_tensor(tensor)

        assert fig is not None
        plt.close(fig)

    def test_visualize_tensor_requires_2d(self):
        """Test that visualize_tensor rejects non-2D tensors."""
        tensor_3d = np.random.randn(5, 5, 5)

        with pytest.raises(ValueError):
            visualization.visualize_tensor(tensor_3d)

    def test_display_model_summary(self, capsys):
        """Test model summary display."""
        layers = [
            {"name": "conv1", "type": "Conv2D", "output_shape": "(N, 32, 28, 28)", "params": 1000},
            {"name": "relu1", "type": "ReLU", "output_shape": "(N, 32, 28, 28)", "params": 0},
            {"name": "fc1", "type": "Linear", "output_shape": "(N, 10)", "params": 1290},
        ]

        visualization.display_model_summary(layers)

        captured = capsys.readouterr()
        assert "conv1" in captured.out
        assert "Total parameters: 2,290" in captured.out

    def test_plot_class_distribution(self):
        """Test class distribution plotting."""
        import matplotlib.pyplot as plt

        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        fig = visualization.plot_class_distribution(labels)

        assert fig is not None
        plt.close(fig)

    def test_plot_class_distribution_with_names(self):
        """Test class distribution with class names."""
        import matplotlib.pyplot as plt

        labels = np.array([0, 0, 1, 1, 2, 2])
        class_names = ["Cat", "Dog", "Bird"]

        fig = visualization.plot_class_distribution(labels, class_names)

        assert fig is not None
        plt.close(fig)


class TestImports:
    """Test that all utilities can be imported."""

    def test_import_from_init(self):
        """Test importing from notebooks.utils __init__."""
        from notebooks.utils import (
            run_mojo_script,
            compile_mojo_binary,
            numpy_to_mojo_binary,
            mojo_binary_to_numpy,
            plot_training_curves,
            plot_confusion_matrix,
            visualize_tensor,
            TrainingProgressBar,
        )

        # Just verify they're imported
        assert callable(run_mojo_script)
        assert callable(compile_mojo_binary)
        assert callable(numpy_to_mojo_binary)
        assert callable(mojo_binary_to_numpy)
        assert callable(plot_training_curves)
        assert callable(plot_confusion_matrix)
        assert callable(visualize_tensor)
        assert TrainingProgressBar is not None
