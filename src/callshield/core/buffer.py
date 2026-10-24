"""Fixed-capacity rolling buffer for streaming audio samples."""

import numpy as np


class RollingBuffer:
    """Fixed-capacity 1D rolling audio buffer."""

    def __init__(self, capacity_samples: int) -> None:
        if capacity_samples <= 0:
            raise ValueError(f"capacity_samples must be positive, got {capacity_samples}")
        self.capacity_samples = capacity_samples
        self._data: np.ndarray = np.empty(0, dtype=np.float32)

    def append(self, samples: np.ndarray) -> None:
        """Appends samples to the buffer, dropping oldest samples when capacity is exceeded."""
        samples_arr = np.asarray(samples, dtype=np.float32)
        if len(samples_arr) == 0:
            return

        if len(samples_arr) >= self.capacity_samples:
            self._data = samples_arr[-self.capacity_samples :].copy()
        else:
            combined = np.concatenate([self._data, samples_arr])
            if len(combined) > self.capacity_samples:
                self._data = combined[-self.capacity_samples :]
            else:
                self._data = combined

    def latest(self, n: int) -> np.ndarray:
        """Returns the most recent n samples (or fewer if fewer are buffered)."""
        if n <= 0:
            return np.empty(0, dtype=np.float32)
        return self._data[-n:].copy()

    def __len__(self) -> int:
        return len(self._data)
