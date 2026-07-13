"""Progress tracking utilities for training in notebooks.

Uses ipywidgets for interactive progress displays.
"""

from IPython.display import display, HTML
from typing import Optional, Dict
import time


class TrainingProgressBar:
    """Interactive progress bar for training.

    Tracks epochs and batches with live updates.
    """

    def __init__(
        self,
        total_epochs: int,
        total_batches: Optional[int] = None,
        description: str = "Training",
    ):
        """Initialize progress bar.

        Args:
            total_epochs: Total number of epochs to train
            total_batches: Optional total batches per epoch
            description: Description to display
        """
        self.total_epochs = total_epochs
        self.total_batches = total_batches or 100
        self.description = description

        self.current_epoch = 0
        self.current_batch = 0
        self.epoch_losses = []
        self.epoch_accuracies = []
        self.batch_losses = []

        self.start_time = time.time()
        self.epoch_start_time = None

    def start_epoch(self, epoch: int) -> None:
        """Start tracking a new epoch.

        Args:
            epoch: Epoch number (0-indexed)
        """
        self.current_epoch = epoch
        self.current_batch = 0
        self.batch_losses = []
        self.epoch_start_time = time.time()

        epoch_display = epoch + 1  # Display as 1-indexed
        print(f"\nEpoch {epoch_display}/{self.total_epochs}")

    def update_batch(self, batch: int, loss: float) -> None:
        """Update progress for batch completion.

        Args:
            batch: Batch number (0-indexed)
            loss: Loss value for this batch
        """
        self.current_batch = batch
        self.batch_losses.append(loss)

        # Show progress every 10 batches
        if (batch + 1) % 10 == 0:
            avg_loss = sum(self.batch_losses[-10:]) / 10
            progress = (batch + 1) / self.total_batches * 100
            print(f"  Batch {batch + 1}/{self.total_batches} - Loss: {avg_loss:.4f} ({progress:.0f}%)")

    def end_epoch(self, loss: float, accuracy: Optional[float] = None) -> None:
        """Update progress for epoch completion.

        Args:
            loss: Average loss for the epoch
            accuracy: Optional accuracy metric
        """
        self.epoch_losses.append(loss)
        if accuracy is not None:
            self.epoch_accuracies.append(accuracy)

        elapsed = time.time() - self.epoch_start_time
        avg_loss = sum(self.batch_losses) / len(self.batch_losses)

        if accuracy is not None:
            print(f"✓ Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Time: {elapsed:.1f}s")
        else:
            print(f"✓ Loss: {avg_loss:.4f}, Time: {elapsed:.1f}s")

    def finish(self) -> Dict:
        """Complete training and return summary.

        Returns:
            Dict with training statistics
        """
        total_time = time.time() - self.start_time

        return {
            "total_time_seconds": total_time,
            "num_epochs": len(self.epoch_losses),
            "final_loss": self.epoch_losses[-1] if self.epoch_losses else None,
            "best_loss": min(self.epoch_losses) if self.epoch_losses else None,
            "final_accuracy": (self.epoch_accuracies[-1] if self.epoch_accuracies else None),
            "best_accuracy": (max(self.epoch_accuracies) if self.epoch_accuracies else None),
        }

    def plot_history(self):
        """Plot training history (requires matplotlib).

        Returns:
            Matplotlib figure object
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss
        axes[0].plot(self.epoch_losses, "b-", linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].grid(True, alpha=0.3)

        # Plot accuracy if available
        if self.epoch_accuracies:
            axes[1].plot(self.epoch_accuracies, "g-", linewidth=2)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_title("Training Accuracy")
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "No accuracy data", ha="center", va="center")
            axes[1].set_title("Training Accuracy")

        plt.tight_layout()
        return fig


class SimpleProgressBar:
    """Lightweight progress bar without ipywidgets dependency.

    Uses text-based display suitable for notebooks and terminal.
    """

    def __init__(self, total: int, desc: str = "Progress", width: int = 40):
        """Initialize progress bar.

        Args:
            total: Total number of iterations
            desc: Description to display
            width: Width of progress bar in characters
        """
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0

    def update(self, amount: int = 1) -> None:
        """Update progress.

        Args:
            amount: Number of iterations completed
        """
        self.current = min(self.current + amount, self.total)
        self._show()

    def _show(self) -> None:
        """Display progress bar."""
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = "█" * filled + "░" * (self.width - filled)
        print(
            f"\r{self.desc}: |{bar}| {percent * 100:.0f}%",
            end="",
            flush=True,
        )

        if self.current == self.total:
            print()  # Newline when complete

    def close(self) -> None:
        """Close the progress bar."""
        self.current = self.total
        self._show()
