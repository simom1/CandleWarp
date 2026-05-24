from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from numpy.lib.stride_tricks import sliding_window_view

from .core import (
    SimilarityConfig,
    apply_volatility_cuts,
    compute_atr,
    compute_trend_bin,
    compute_volatility_cuts,
    effective_min_match_gap,
)


@dataclass(frozen=True)
class BacktestConfig:
    similarity: SimilarityConfig
    min_history: int = 1000
    stride: int | None = None
    cost_bps: float = 0.0
    edge_threshold_bps: float = 0.0
    max_trials: int | None = None


@dataclass(frozen=True)
class BacktestResult:
    trials: pd.DataFrame
    summary: dict
    buckets: pd.DataFrame
    skipped: int
    skip_reasons: dict


def run_walk_forward(df: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    sim = config.similarity
    stride = config.stride or sim.horizon
    start_index = max(
        config.min_history,
        sim.window + sim.horizon + sim.atr_period + max(sim.trend_lookback, 0),
    )
    stop_index = len(df) - sim.horizon
    if stop_index <= start_index:
        raise ValueError("Not enough rows for walk-forward evaluation.")

    working = prepare_walk_forward_frame(df, sim, start_index)
    features = make_feature_matrix(working, sim.window, sim.body_ratio_weight)
    terminal_returns = make_terminal_returns(working["close"].to_numpy(dtype=float), sim.horizon)

    rows: list[dict] = []
    skipped = 0
    skip_reasons: dict[str, int] = {
        "invalid_state": 0,
        "not_enough_history": 0,
        "not_enough_candidates": 0,
        "not_enough_diverse": 0,
    }

    for query_end in range(start_index, stop_index, stride):
        if config.max_trials is not None and len(rows) >= config.max_trials:
            break

        query_vector = features[query_end]
        query_state = working.loc[query_end, "state"]
        if not np.isfinite(query_vector).all() or "unknown" in str(query_state):
            skip_reasons["invalid_state"] += 1
            skipped += 1
            continue

        query_start = query_end - sim.window + 1
        candidate_end_max = min(query_start - 1, query_end - sim.horizon)
        if candidate_end_max < sim.window - 1:
            skip_reasons["not_enough_history"] += 1
            skipped += 1
            continue

        candidate_indices = np.arange(sim.window - 1, candidate_end_max + 1)
        valid = (
            np.isfinite(features[candidate_indices]).all(axis=1)
            & np.isfinite(terminal_returns[candidate_indices])
            & (working.loc[candidate_indices, "state"].to_numpy() == query_state)
        )
        candidate_indices = candidate_indices[valid]
        if len(candidate_indices) < sim.top_k:
            skip_reasons["not_enough_candidates"] += 1
            skipped += 1
            continue

        distances = np.linalg.norm(features[candidate_indices] - query_vector, axis=1)
        top_indices = select_diverse_indices(
            candidate_indices,
            distances,
            sim.top_k,
            effective_min_match_gap(sim),
        )
        if len(top_indices) < sim.top_k:
            skip_reasons["not_enough_diverse"] += 1
            skipped += 1
            continue

        top_returns = terminal_returns[top_indices]
        state_returns = terminal_returns[candidate_indices]
        sim_terminal = quantile_dict(top_returns)
        baseline = quantile_dict(state_returns)
        baseline["count"] = int(len(state_returns))
        actual_return = float(terminal_returns[query_end])
        side = trade_side(
            sim_terminal["q25"],
            sim_terminal["q75"],
            threshold=config.edge_threshold_bps / 10000.0,
        )
        net_return = side * actual_return
        if side != 0:
            net_return -= config.cost_bps / 10000.0

        rows.append(
            {
                "query_end_index": query_end,
                "query_end_time": working.loc[query_end, "timestamp"],
                "state": query_state,
                "matches": int(len(top_indices)),
                "actual_return": actual_return,
                "sim_q10": sim_terminal["q10"],
                "sim_q25": sim_terminal["q25"],
                "sim_median": sim_terminal["q50"],
                "sim_q75": sim_terminal["q75"],
                "sim_q90": sim_terminal["q90"],
                "state_q25": baseline["q25"],
                "state_median": baseline["q50"],
                "state_q75": baseline["q75"],
                "state_candidates": baseline["count"],
                "sim_median_abs_error": abs(actual_return - sim_terminal["q50"]),
                "state_median_abs_error": abs(actual_return - baseline["q50"]),
                "covered_25_75": sim_terminal["q25"] <= actual_return <= sim_terminal["q75"],
                "covered_10_90": sim_terminal["q10"] <= actual_return <= sim_terminal["q90"],
                "direction_hit": same_sign(sim_terminal["q50"], actual_return),
                "state_direction_hit": same_sign(baseline["q50"], actual_return),
                "trade_side": side,
                "strategy_return": net_return,
            }
        )

    trials = pd.DataFrame(rows)
    if trials.empty:
        raise ValueError("No valid walk-forward trials. Try reducing --min-history or --top-k.")

    buckets = make_bucket_table(trials)
    summary = summarize_trials(trials, config, skipped)
    summary["skip_reasons"] = skip_reasons
    return BacktestResult(
        trials=trials,
        summary=summary,
        buckets=buckets,
        skipped=skipped,
        skip_reasons=skip_reasons,
    )


def prepare_walk_forward_frame(
    df: pd.DataFrame,
    config: SimilarityConfig,
    first_query_index: int,
) -> pd.DataFrame:
    working = df.copy().reset_index(drop=True)
    working["atr"] = compute_atr(working, config.atr_period)
    working["trend_bin"] = compute_trend_bin(
        working["close"],
        lookback=config.trend_lookback,
        flat_threshold=config.flat_threshold,
    )

    atr_pct = (working["atr"] / working["close"]).replace([np.inf, -np.inf], np.nan)
    # Fix: freeze volatility thresholds at the first query point to avoid lookahead leakage.
    # Using full-sample quantiles would let future vol regimes influence historical bin labels.
    cuts = compute_volatility_cuts(atr_pct.iloc[: first_query_index + 1])
    working["vol_bin"] = apply_volatility_cuts(atr_pct, cuts)

    working["state"] = working["trend_bin"] + "/" + working["vol_bin"]
    return working


def make_feature_matrix(
    df: pd.DataFrame, window: int, body_ratio_weight: float = 3.0
) -> np.ndarray:
    """Build a (n_rows, window*5) feature matrix via vectorised sliding windows.

    Each row corresponds to the window ending at that bar index.  Rows without a
    complete valid window are left as NaN.  Uses numpy sliding_window_view for a
    zero-copy strided view, replacing the previous O(n) Python loop.
    """
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)
    n = len(df)
    features = np.full((n, window * 5), np.nan, dtype=float)
    if n < window:
        return features

    # sliding_window_view: shape (n - window + 1, window) – zero-copy strided view
    open_w = sliding_window_view(open_, window)
    high_w = sliding_window_view(high, window)
    low_w = sliding_window_view(low, window)
    close_w = sliding_window_view(close, window)

    ends = np.arange(window - 1, n)   # end bar index for each window
    denom = atr[ends]                  # ATR at last bar of each window
    ref = close_w[:, 0]               # reference close = first bar of each window

    candle_range = high_w - low_w
    body_ratio = (
        np.where(candle_range != 0, (close_w - open_w) / candle_range, 0.0)
        * body_ratio_weight
    )
    rel_open = (open_w - ref[:, None]) / denom[:, None]
    rel_high = (high_w - ref[:, None]) / denom[:, None]
    rel_low = (low_w - ref[:, None]) / denom[:, None]
    rel_close = (close_w - ref[:, None]) / denom[:, None]

    # Stack → (n_windows, window, 5) → flatten to (n_windows, window*5)
    encoded = np.stack(
        [rel_open, rel_high, rel_low, rel_close, body_ratio], axis=2
    ).reshape(len(ends), -1)

    valid = np.isfinite(denom) & (denom > 0) & np.isfinite(encoded).all(axis=1)
    features[ends[valid]] = encoded[valid]
    return features


def make_terminal_returns(close: np.ndarray, horizon: int) -> np.ndarray:
    returns = np.full(len(close), np.nan, dtype=float)
    returns[: len(close) - horizon] = close[horizon:] / close[: len(close) - horizon] - 1.0
    return returns


def quantile_dict(values: np.ndarray) -> dict:
    q10, q25, q50, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "q10": float(q10),
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "q90": float(q90),
    }


def select_diverse_indices(
    candidate_indices: np.ndarray,
    distances: np.ndarray,
    top_k: int,
    min_gap: int,
) -> np.ndarray:
    order = np.argsort(distances)
    sorted_candidates = candidate_indices[order]
    if min_gap <= 0:
        return sorted_candidates[:top_k]

    # Pre-allocate result buffer; iterate sorted candidates once.
    # Inner check is O(n_selected) ≤ O(top_k) which is small, so overall O(n).
    buffer = np.empty(top_k, dtype=int)
    n_sel = 0
    for candidate in sorted_candidates:
        if n_sel == 0 or np.all(np.abs(candidate - buffer[:n_sel]) >= min_gap):
            buffer[n_sel] = candidate
            n_sel += 1
            if n_sel >= top_k:
                break
    return buffer[:n_sel]


def terminal_quantiles(quantiles: pd.DataFrame, horizon: int) -> dict:
    column = f"t+{horizon}"
    values = quantiles.set_index("quantile")[column]
    return {
        "q10": float(values.loc[0.10]),
        "q25": float(values.loc[0.25]),
        "q50": float(values.loc[0.50]),
        "q75": float(values.loc[0.75]),
        "q90": float(values.loc[0.90]),
    }


def trade_side(q25: float, q75: float, threshold: float) -> int:
    if q25 > threshold:
        return 1
    if q75 < -threshold:
        return -1
    return 0


def same_sign(left: float, right: float) -> bool:
    if left == 0 or right == 0:
        return False
    return np.sign(left) == np.sign(right)


def make_bucket_table(trials: pd.DataFrame) -> pd.DataFrame:
    unique_values = trials["sim_median"].nunique()
    bucket_count = min(5, unique_values)
    if bucket_count < 2:
        return pd.DataFrame()

    bucketed = trials.copy()
    bucketed["bucket"] = pd.qcut(bucketed["sim_median"], q=bucket_count, duplicates="drop")
    grouped = (
        bucketed.groupby("bucket", observed=True)
        .agg(
            count=("actual_return", "size"),
            avg_sim_median=("sim_median", "mean"),
            avg_actual_return=("actual_return", "mean"),
            median_actual_return=("actual_return", "median"),
            hit_rate=("direction_hit", "mean"),
        )
        .reset_index()
    )
    grouped["bucket"] = grouped["bucket"].astype(str)
    return grouped


def summarize_trials(trials: pd.DataFrame, config: BacktestConfig, skipped: int) -> dict:
    sim_mae = float(trials["sim_median_abs_error"].mean())
    state_mae = float(trials["state_median_abs_error"].mean())
    improvement = state_mae - sim_mae
    improvement_pct = improvement / state_mae if state_mae > 0 else np.nan
    spearman_ic = trials[["sim_median", "actual_return"]].corr(method="spearman").iloc[0, 1]

    trade_rows = trials[trials["trade_side"] != 0].copy()
    trade_summary = summarize_trades(trade_rows)
    trade_summary["trade_rate"] = float(len(trade_rows) / len(trials))

    return {
        "timeframe": config.similarity.timeframe,
        "window": config.similarity.window,
        "horizon": config.similarity.horizon,
        "top_k": config.similarity.top_k,
        "min_history": config.min_history,
        "stride": config.stride or config.similarity.horizon,
        "cost_bps": config.cost_bps,
        "edge_threshold_bps": config.edge_threshold_bps,
        "trials": int(len(trials)),
        "skipped": int(skipped),
        "coverage_25_75": float(trials["covered_25_75"].mean()),
        "coverage_10_90": float(trials["covered_10_90"].mean()),
        "direction_hit_rate": float(trials["direction_hit"].mean()),
        "state_direction_hit_rate": float(trials["state_direction_hit"].mean()),
        "median_mae_similarity": sim_mae,
        "median_mae_state_baseline": state_mae,
        "median_mae_improvement": float(improvement),
        "median_mae_improvement_pct": float(improvement_pct),
        "spearman_ic": float(spearman_ic) if pd.notna(spearman_ic) else np.nan,
        **trade_summary,
    }


def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "trade_rate": 0.0,
            "avg_trade_return": np.nan,
            "trade_win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown": np.nan,
            "return_to_risk": np.nan,
            "compounded_return": 0.0,
        }

    returns = trades["strategy_return"].astype(float)
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    std = returns.std(ddof=1)
    return_to_risk = returns.mean() / std if std and std > 0 else np.nan

    return {
        "trades": int(len(trades)),
        "trade_rate": np.nan,
        "avg_trade_return": float(returns.mean()),
        "trade_win_rate": float((returns > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else np.inf,
        "max_drawdown": float(drawdown.min()),
        "return_to_risk": float(return_to_risk) if pd.notna(return_to_risk) else np.nan,
        "compounded_return": float(equity.iloc[-1] - 1.0),
    }
