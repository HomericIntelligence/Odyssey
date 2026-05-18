# ML Odyssey Training Dashboard

Flask-based web dashboard for monitoring training runs.

## Features

- Real-time visualization of training metrics
- Compare multiple runs side-by-side
- Auto-refresh every 30 seconds
- Interactive Plotly.js charts
- Dark theme optimized for readability

## Quick Start

```bash
# Start the dashboard server
pixi run python scripts/dashboard/server.py

# Open in browser
# http://127.0.0.1:5000
```

## Command Line Options

```bash
pixi run python scripts/dashboard/server.py --help

Options:
  --port PORT       Port to run server on (default: 5000)
  --host HOST       Host to bind to (default: 127.0.0.1)
  --logs-dir DIR    Directory containing training logs (default: logs/)
  --debug           Run in debug mode with auto-reload
```

## How It Works

The dashboard reads CSV files from the logs directory. Each training run should
be in its own subdirectory with CSV files for each metric.

Expected structure:

```text
logs/
  lenet5_run1/
    train_loss.csv
    train_accuracy.csv
  lenet5_run2/
    train_loss.csv
    train_accuracy.csv
```

CSV format (compatible with CSVMetricsLogger):

```csv
step,value
0,0.5
1,0.45
2,0.4
```

## API Endpoints

| Endpoint | Description |
| --- | --- |
| `GET /` | Dashboard UI |
| `GET /api/runs` | List all training runs |
| `GET /api/run/<id>` | Get metadata for a run |
| `GET /api/run/<id>/metrics` | Get all metrics for a run |
| `GET /api/run/<id>/metric/<name>` | Get specific metric data |
| `GET /api/compare?runs=a,b&metric=loss` | Compare runs |

## Integration with Training

Use `CSVMetricsLogger` from `shared.training.metrics` to log metrics:

```mojo
from projectodyssey.training.metrics import CSVMetricsLogger

var logger = CSVMetricsLogger("logs/my_run")
logger.log_scalar("train_loss", 0.5)
logger.log_scalar("train_accuracy", 0.85)
logger.step()
_ = logger.save()
```
