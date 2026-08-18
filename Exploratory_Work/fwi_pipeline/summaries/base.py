from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class SummaryMetric(ABC):
    @abstractmethod
    def compute(self, cube, region_cfg: dict) -> pd.DataFrame:
        """Return a DataFrame with columns [Date, FWI] (or similar) for the
        given iris cube, constrained/aggregated per region_cfg."""
