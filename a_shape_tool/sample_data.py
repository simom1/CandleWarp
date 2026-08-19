from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_sample_ohlc(output_path: str | Path, rows: int = 900, seed: int = 7) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=rows, freq="h")

    close = np.empty(rows)
    open_ = np.empty(rows)
    high = np.empty(rows)
    low = np.empty(rows)
    close[0] = 2050.0
    open_[0] = close[0]

    regime_drifts = np.array([0.00018, -0.00012, 0.0, 0.00028, -0.00022])
    regime_vols = np.array([0.0018, 0.0023, 0.0011, 0.0032, 0.0028])

    for i in range(1, rows):
        regime = (i // 140) % len(regime_drifts)
        wave = 0.00045 * np.sin(i / 17.0)
        ret = regime_drifts[regime] + wave + rng.normal(0.0, regime_vols[regime])
        open_[i] = close[i - 1] * (1.0 + rng.normal(0.0, regime_vols[regime] * 0.12))
        close[i] = open_[i] * np.exp(ret)

    volume = np.empty(rows)
    for i in range(rows):
        regime = (i // 140) % len(regime_vols)
        base_range = close[i] * abs(rng.normal(regime_vols[regime] * 1.4, regime_vols[regime] * 0.45))
        upper = base_range * rng.uniform(0.35, 0.9)
        lower = base_range * rng.uniform(0.35, 0.9)
        high[i] = max(open_[i], close[i]) + upper
        low[i] = min(open_[i], close[i]) - lower
        # Volume correlated with range and volatility regime
        vol_intensity = (high[i] - low[i]) / (close[i] * regime_vols[regime] + 1e-8)
        volume[i] = max(100.0, rng.lognormal(mean=8.0, sigma=0.45) * vol_intensity)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    df.to_csv(output_path, index=False, float_format="%.6f")
    return output_path

