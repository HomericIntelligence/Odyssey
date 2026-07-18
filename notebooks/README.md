# ML Odyssey Jupyter Notebooks

Interactive Jupyter notebooks for exploring, learning, and experimenting with ML Odyssey models
and techniques.

## Quick Start

Launch Jupyter Lab to explore the notebooks:

```bash
# Launch Jupyter Lab
just jupyter

# Or use classic notebook interface
just jupyter-notebook
```

Then navigate to `http://localhost:8888` in your browser.

## Available Notebooks

| # | Notebook | Description | Prerequisites |
| --- | --- | --- | --- |
| 1 | `01_introduction` | Quick start and overview | None |
| 2 | `02_tensor_operations` | Tensor fundamentals | None |
| 3 | `03_building_models` | Model architecture design | Notebook 2 |
| 4 | `04_training_mnist` | Complete training pipeline | EMNIST dataset* |
| 5 | `05_visualization` | Plotting and analysis | Trained models |
| 6 | `06_advanced_techniques` | Mixed precision & optimization | None |

*EMNIST dataset is automatically downloaded on first run.

## Setup

### 1. Install Dependencies

All dependencies are already configured in `pyproject.toml`. Just sync the uv environment (and prefix commands with `uv run`, or activate `.venv`):

```bash
uv sync --locked
```

### 2. Optional: Download Datasets (for training notebooks)

The training notebook will auto-download EMNIST on first run. To pre-download:

```bash
# EMNIST will be cached in ~/.cache/ml-odyssey/
# The notebook handles this automatically
```

### 3. Launch Notebooks

```bash
just jupyter
```

## Architecture

The notebook infrastructure bridges Python and Mojo:

```text
┌─────────────────────────────┐
│  Jupyter Notebook (Python)  │
└──────────────┬──────────────┘
               │
      ┌────────▼────────┐
      │ notebooks/utils │
      ├────────┬────────┤
      │  Python Libraries  │
      ├─────────────────────┤
      │ • matplotlib       │
      │ • numpy            │
      │ • seaborn          │
      │ • ipywidgets       │
      └─────────────────────┘
               │
      ┌────────▼──────────────┐
      │  Mojo Bridge Calls    │
      │ (subprocess + uv)     │
      └────────┬──────────────┘
               │
      ┌────────▼──────────────────┐
      │  Compiled Mojo Binaries   │
      │  • Training executables   │
      │  • Inference engines      │
      │  • Tensor operations      │
      └───────────────────────────┘
```

## Utilities

The `notebooks/utils/` package provides helpers:

### `mojo_bridge.py`

- Run Mojo scripts from notebooks
- Compile Mojo code to binaries
- Parse JSON/structured output

### `tensor_utils.py`

- Convert between NumPy and Mojo binary formats
- Save/load tensor metadata
- Compare tensors for debugging

### `visualization.py`

- Plot training curves (loss, accuracy)
- Display confusion matrices
- Visualize tensor heatmaps
- Show model architecture summary

### `progress.py`

- Track training progress with live updates
- Display epoch/batch statistics
- Generate training history plots

## Usage Examples

### Load and Run a Mojo Script

```python
from notebooks.utils import run_mojo_script

result = run_mojo_script("examples/my_model/train.mojo", args=["--epochs", "10"])
print(result['stdout'])
```

### Convert Tensors

```python
from notebooks.utils import numpy_to_mojo_binary, mojo_binary_to_numpy
import numpy as np

# Save NumPy array for Mojo to load
array = np.random.randn(10, 10).astype(np.float32)
numpy_to_mojo_binary(array, "tensor.bin")

# Load Mojo-generated binary back to NumPy
loaded = mojo_binary_to_numpy("output.bin", shape=(10, 10))
```

### Plot Training Results

```python
from notebooks.utils import plot_training_curves

losses = [1.2, 1.0, 0.8, 0.6, 0.4]
fig = plot_training_curves(losses)
plt.show()
```

### Track Training Progress

```python
from notebooks.utils import TrainingProgressBar

progress = TrainingProgressBar(total_epochs=10, total_batches=100)

for epoch in range(10):
    progress.start_epoch(epoch)
    for batch in range(100):
        loss = ...  # computed loss
        progress.update_batch(batch, loss)
    progress.end_epoch(avg_loss)

summary = progress.finish()
progress.plot_history()
```

## Cell-by-Cell Execution

Notebooks are designed to be run incrementally. Each cell:

1. Clearly shows what it's doing
2. Prints status messages
3. Saves outputs for next cells to use
4. Has error handling with helpful messages

You can:

- Run individual cells with Shift+Enter
- Run all cells with Kernel → Restart & Run All
- Skip cells (e.g., skip long training to load pre-trained weights)

## Saving Outputs

Notebooks should have outputs cleared before committing. Use:

```bash
# Clear all notebook outputs
just jupyter-clear

# Or clear specific notebook
jupyter nbconvert --clear-output --inplace notebooks/01_introduction.ipynb
```

The pre-commit hook `nbstripout` will strip outputs automatically before commits.

## Testing Notebooks

Validate all notebooks execute without errors:

```bash
# Run validation (non-training notebooks only)
just jupyter-validate

# This skips notebooks that require external datasets
# to keep CI fast
```

## Troubleshooting

### "Mojo not found" Error

```python
FileNotFoundError: Script not found or mojo not in PATH
```

**Solution**: Ensure the uv env is active (`source .venv/bin/activate` or prefix with `uv run`):

```bash
source .venv/bin/activate
just jupyter
```

### "Timeout" on Training Notebook

```text
subprocess.TimeoutExpired: Timeout after 300s
```

**Solution**: Increase timeout for training:

```python
run_mojo_script("train.mojo", timeout=600)  # 10 minutes
```

### Memory Issues with Large Datasets

**Solution**: Reduce batch size:

```python
result = run_mojo_script(
    "train.mojo",
    args=["--batch-size", "16"]  # Smaller batches
)
```

### Notebook Kernel Crashes

If the notebook kernel crashes:

1. Click "Kernel" → "Restart"
2. Run cells from top to rebuild state
3. Check for infinite loops or excessive memory use

## Advanced: Custom Notebooks

To create your own notebook:

1. Copy `01_introduction.ipynb` as a template
2. Replace content with your code
3. Use utilities from `notebooks.utils`
4. Clear outputs before committing

Example structure:

```python
# Cell 1: Imports
from notebooks.utils import *
import numpy as np

# Cell 2: Load data
data = ...

# Cell 3: Run model
result = run_mojo_script(...)

# Cell 4: Visualize
plot_training_curves(...)
```

## Integration with CI/CD

Notebooks are validated in CI:

1. **Syntax Check**: All `.ipynb` files have valid JSON
2. **Size Check**: No notebook exceeds 500KB
3. **Pre-commit**: Outputs stripped before commit
4. **Validation**: Quick notebooks execute without error

See `.github/workflows/notebook-validation.yml` for details.

## Future Enhancements

When Mojo gains Jupyter kernel support, notebooks can:

- Include native Mojo cells with syntax highlighting
- Use `%%mojo` magic commands
- Display AnyTensor directly without conversion
- Access Mojo REPL interactively

## Contributing

When adding new notebooks:

1. Follow existing naming: `NN_name.ipynb` (NN = 01, 02, ...)
2. Add to the table above
3. Include clear markdown descriptions
4. Clear outputs: `just jupyter-clear`
5. Test execution: `just jupyter-validate`
6. Link in main `README.md`

## References

- [Jupyter Documentation](https://jupyter.org/)
- [Mojo Documentation](https://docs.modular.com/mojo/)
- [ML Odyssey Docs](/docs/)
- [Example Models](/examples/)
