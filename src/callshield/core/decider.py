"""Sliding-window decision logic and latching for Call Shield."""

import math
from collections import deque


class AlertDecider:
    """Decision maker tracking sliding-window suspicious scores and alert latching.

    Conventions:
      - Suspicious window: score < thr
      - History: last n boolean outcomes
      - Flagged count: sum of suspicious windows in history
      - Level progression:
          * flagged >= k -> ALERT and LATCHES for session
          * flagged >= k - 2 -> WARNING
          * otherwise -> OK
      - NaN scores: treated as 'no information', ignored in history
    """

    def __init__(
        self,
        thr: float = 0.0,
        k: int = 4,
        n: int = 6,
        auto_recover: bool = False,
        recovery_consecutive_ok: int = 6,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if n < k:
            raise ValueError(f"n must be >= k, got n={n}, k={k}")

        self.thr = float(thr)
        self.k = int(k)
        self.n = int(n)
        self.auto_recover = bool(auto_recover)
        self.recovery_consecutive_ok = int(recovery_consecutive_ok)
        self.consecutive_genuine: int = 0
        self.history: deque[bool] = deque(maxlen=n)
        self.latched: bool = False

    @property
    def flagged(self) -> int:
        """Returns the count of suspicious windows currently in history."""
        return sum(self.history)

    def update(self, score: float) -> tuple[str, int, str]:
        """Evaluates a new score and returns (level, flagged, reason)."""
        if math.isnan(score):
            current_level = (
                "ALERT" if self.latched else ("WARNING" if self.flagged >= self.k - 2 else "OK")
            )
            return current_level, self.flagged, "nan_score"

        is_suspicious = bool(score < self.thr)
        self.history.append(is_suspicious)

        if not is_suspicious:
            self.consecutive_genuine += 1
        else:
            self.consecutive_genuine = 0

        # Auto-recovery if enabled and sufficient consecutive genuine scores observed
        if self.auto_recover and self.latched and self.consecutive_genuine >= self.recovery_consecutive_ok:
            self.latched = False

        current_flagged = self.flagged
        if current_flagged >= self.k:
            self.latched = True

        if self.latched:
            level = "ALERT"
        elif current_flagged >= self.k - 2:
            level = "WARNING"
        else:
            level = "OK"

        return level, current_flagged, "scored"

    def reset(self) -> None:
        """Resets the history queue and clears the alert latch."""
        self.history.clear()
        self.latched = False
        self.consecutive_genuine = 0
