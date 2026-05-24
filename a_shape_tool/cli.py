from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import SimilarityConfig, load_ohlc_csv, run_similarity
from .sample_data import make_sample_ohlc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Target-A historical OHLC shape similarity distribution demo."
    )
    parser.add_argument("--csv", help="Input OHLC CSV path.")
    parser.add_argument("--make-sample", help="Write deterministic sample OHLC CSV to this path.")
    parser.add_argument("--rows", type=int, default=900, help="Rows for --make-sample.")
    parser.add_argument(
        "--timeframe",
        default="1h",
        help="Timeframe label for reports. Default: 1h.",
    )
    parser.add_argument("--window", type=int, default=100, help="Shape matching window length.")
    parser.add_argument("--horizon", type=int, default=50, help="Forward bars to visualize.")
    parser.add_argument("--top-k", type=int, default=50, help="Number of most similar windows.")
    parser.add_argument(
        "--body-ratio-weight",
        type=float,
        default=3.0,
        help=(
            "Scale factor for the body_ratio feature (candle body / candle range ∈ [-1,1]). "
            "ATR-normalised features span a much larger range; this weight balances their "
            "contribution to the Euclidean distance. Default: 3.0."
        ),
    )
    parser.add_argument(
        "--min-match-gap",
        type=int,
        help="Minimum bars between selected match window ends. Default: max(window, horizon).",
    )
    parser.add_argument("--atr-period", type=int, default=14, help="ATR normalization period.")
    parser.add_argument(
        "--trend-lookback",
        type=int,
        default=120,
        help="Lookback bars for high-level trend state. Use 0 to disable trend split.",
    )
    parser.add_argument(
        "--flat-threshold",
        type=float,
        default=0.0,
        help="Absolute return threshold for flat trend bin, e.g. 0.003.",
    )
    parser.add_argument(
        "--query-end",
        default="last",
        help="Query window end: 'last', row index, negative row index, or timestamp.",
    )
    parser.add_argument(
        "--no-history-only",
        action="store_true",
        help="Allow candidates after the query window for research playback.",
    )
    parser.add_argument(
        "--no-dtw",
        action="store_true",
        help="Disable Dynamic Time Warping (DTW) and use pure Euclidean distance.",
    )
    parser.add_argument(
        "--dtw-warping-window",
        type=int,
        default=10,
        help="DTW Sakoe-Chiba warping window. Default: 10.",
    )
    parser.add_argument(
        "--dtw-rerank-k",
        type=int,
        default=200,
        help="Number of Euclidean pre-filtered candidates to re-rank with DTW. Default: 200.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run walk-forward evaluation instead of one latest-window report.",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=1000,
        help="Minimum known bars before the first walk-forward trial.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        help="Bars between walk-forward trials. Default: same as --horizon.",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=0.0,
        help="Round-trip cost in basis points for the optional strategy rule.",
    )
    parser.add_argument(
        "--edge-threshold-bps",
        type=float,
        default=0.0,
        help="Required q25/q75 edge in basis points before taking a trade.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        help="Limit walk-forward trials for quick research runs.",
    )
    parser.add_argument("--output-dir", default="output", help="Output directory.")
    # ── most-common mode ──
    parser.add_argument(
        "--most-common",
        action="store_true",
        help="Find the most frequently recurring window shapes in history instead of querying the latest window.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[50, 100, 200],
        metavar="W",
        help="One or more window sizes to analyse in --most-common mode. Default: 50 100 200.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="Number of distinct archetypes to extract per window size. Default: 3.",
    )
    parser.add_argument(
        "--state-filter",
        default=None,
        help="Trend/volatility state string to restrict K-Means clustering (e.g. 'down/highvol').",
    )
    parser.add_argument(
        "--optimize-clusters",
        action="store_true",
        help="Run K-Means elbow curve optimization for K in [2, 8] and plot results.",
    )
    args = parser.parse_args()

    if args.make_sample:
        sample_path = make_sample_ohlc(args.make_sample, rows=args.rows)
        print(f"Wrote sample OHLC CSV: {sample_path}")
        if not args.csv:
            return

    if not args.csv:
        parser.error("--csv is required unless only --make-sample is used.")

    config = SimilarityConfig(
        timeframe=args.timeframe,
        window=args.window,
        horizon=args.horizon,
        top_k=args.top_k,
        min_match_gap=args.min_match_gap,
        atr_period=args.atr_period,
        trend_lookback=args.trend_lookback,
        flat_threshold=args.flat_threshold,
        query_end=args.query_end,
        history_only=not args.no_history_only,
        body_ratio_weight=args.body_ratio_weight,
        use_dtw=not args.no_dtw,
        dtw_warping_window=args.dtw_warping_window,
        dtw_rerank_k=args.dtw_rerank_k,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_ohlc_csv(args.csv)

    if args.most_common:
        from .common import run_most_common, optimize_kmeans_clusters
        from .plotting import plot_common_patterns, plot_ohlc_candles_grid, plot_elbow_curve, plot_hourly_micro_analysis

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.optimize_clusters:
            print(f"Running K-Means cluster optimization (Elbow Method) | windows={args.windows} | state_filter={args.state_filter}")
            for win in args.windows:
                win_config = SimilarityConfig(
                    timeframe=args.timeframe,
                    window=win,
                    horizon=args.horizon,
                    top_k=args.top_k,
                    atr_period=args.atr_period,
                    trend_lookback=args.trend_lookback,
                    flat_threshold=args.flat_threshold,
                    body_ratio_weight=args.body_ratio_weight,
                    history_only=True,
                    query_end="last",
                )
                try:
                    inertias = optimize_kmeans_clusters(df, win_config, state_filter=args.state_filter)
                except ValueError as exc:
                    print(f"  window={win}: skipped — {exc}")
                    continue

                out_elbow = output_dir / f"common_w{win}_elbow.png"
                plot_elbow_curve(inertias, out_elbow)
                print(f"  window={win}: inertias={inertias} → {out_elbow}")
            return

        print(f"Most-common pattern analysis | windows={args.windows} | n_clusters={args.n_clusters} | state_filter={args.state_filter}")

        for win in args.windows:
            win_config = SimilarityConfig(
                timeframe=args.timeframe,
                window=win,
                horizon=args.horizon,
                top_k=args.top_k,
                atr_period=args.atr_period,
                trend_lookback=args.trend_lookback,
                flat_threshold=args.flat_threshold,
                body_ratio_weight=args.body_ratio_weight,
                history_only=True,
                query_end="last",
            )
            try:
                results = run_most_common(df, win_config, n_clusters=args.n_clusters, state_filter=args.state_filter)
            except ValueError as exc:
                print(f"  window={win}: skipped — {exc}")
                continue

            out_img = output_dir / f"common_w{win}.png"
            plot_common_patterns(
                df, results, out_img,
                title_suffix=f"XAUUSD {args.timeframe} | window={win}" + (f" | state={args.state_filter}" if args.state_filter else ""),
            )
            print(f"  window={win}: {len(results)} clusters → {out_img}")
            for r in results:
                tr = r.terminal_returns
                med = float(r.quantiles.set_index('quantile').loc[0.50, f't+{args.horizon}']) * 100
                up  = float((tr > 0).mean() * 100)
                print(f"    Cluster {r.cluster_id}: count={r.frequency_count} ({r.frequency_pct:.2f}%)  "
                      f"median_h{args.horizon}={med:+.2f}%  up_h{args.horizon}={up:.0f}%")

                # Generate hourly micro-analysis plot for T+1 to T+24
                out_h24 = output_dir / f"common_w{win}_cluster{r.cluster_id}_h24.png"
                plot_title = f"Cluster {r.cluster_id} Micro-Analysis | Window {win} (count={r.frequency_count}, {r.frequency_pct:.1f}%)"
                plot_hourly_micro_analysis(r.hourly_stats, out_h24, plot_title)
                print(f"      Hourly 24h analysis → {out_h24}")

                # Generate a raw candlestick grid for each cluster!
                out_ohlc = output_dir / f"common_w{win}_cluster{r.cluster_id}_ohlc.png"
                plot_ohlc_candles_grid(
                    df,
                    r.member_indices,
                    win,
                    args.horizon,
                    out_ohlc,
                    title=f"Common w={win} | Cluster {r.cluster_id} representative members (horizon={args.horizon})"
                )
                print(f"      Candlestick grid → {out_ohlc}")
        return

    if args.backtest:
        from .evaluation import BacktestConfig, run_walk_forward

        backtest = run_walk_forward(
            df,
            BacktestConfig(
                similarity=config,
                min_history=args.min_history,
                stride=args.stride,
                cost_bps=args.cost_bps,
                edge_threshold_bps=args.edge_threshold_bps,
                max_trials=args.max_trials,
            ),
        )
        trials_path = output_dir / "backtest_trials.csv"
        buckets_path = output_dir / "backtest_buckets.csv"
        summary_path = output_dir / "backtest_summary.json"
        backtest.trials.to_csv(trials_path, index=False)
        backtest.buckets.to_csv(buckets_path, index=False)
        summary_path.write_text(
            json.dumps(backtest.summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        summary = backtest.summary
        print("Walk-forward evaluation done.")
        print(f"Trials: {summary['trials']} (skipped {summary['skipped']})")
        print(f"Median MAE, similarity: {summary['median_mae_similarity'] * 100.0:.3f}%")
        print(f"Median MAE, state baseline: {summary['median_mae_state_baseline'] * 100.0:.3f}%")
        print(f"MAE improvement: {summary['median_mae_improvement_pct'] * 100.0:.2f}%")
        print(f"Spearman IC: {summary['spearman_ic']:.4f}")
        print(f"25%-75% coverage: {summary['coverage_25_75'] * 100.0:.2f}%")
        print(f"10%-90% coverage: {summary['coverage_10_90'] * 100.0:.2f}%")
        print(f"Trades from conservative q25/q75 rule: {summary['trades']}")
        if summary["trades"] > 0:
            print(f"Average trade return: {summary['avg_trade_return'] * 100.0:.3f}%")
            print(f"Trade win rate: {summary['trade_win_rate'] * 100.0:.2f}%")
            print(f"Max drawdown: {summary['max_drawdown'] * 100.0:.2f}%")
        skip_reasons = backtest.skip_reasons
        if any(skip_reasons.values()):
            reasons_str = ", ".join(f"{k}={v}" for k, v in skip_reasons.items() if v > 0)
            print(f"Skip reasons: {reasons_str}")
        print(f"Trials: {trials_path}")
        print(f"Buckets: {buckets_path}")
        print(f"Summary: {summary_path}")
        return

    result = run_similarity(df, config)

    from .plotting import plot_distribution, plot_similarity_diagnostics, write_html_report, plot_ohlc_candles_grid

    image_path = plot_distribution(result, output_dir / "distribution.png")
    diagnostics_path = plot_similarity_diagnostics(df, result, output_dir / "similarity_diagnostics.png")
    
    # Generate the premium raw OHLC candlestick plot for the top matches!
    ohlc_path = output_dir / "top_matches_ohlc.png"
    plot_ohlc_candles_grid(
        df,
        result.matches["end_index"].to_numpy(dtype=int),
        config.window,
        config.horizon,
        ohlc_path,
        title=f"Similarity Query vs Top Matches (window={config.window}, horizon={config.horizon})",
        query_index=int(result.query["query_end_index"]),
    )
    
    report_path = write_html_report(result, image_path, output_dir / "report.html", diagnostics_path)
    matches_path = output_dir / "matches.csv"
    paths_path = output_dir / "paths.csv"
    quantiles_path = output_dir / "quantiles.csv"
    result.matches.to_csv(matches_path, index=False)
    result.paths.to_csv(paths_path, index=False)
    result.quantiles.to_csv(quantiles_path, index=False)

    terminal_column = f"t+{config.horizon}"
    terminal = result.quantiles[["quantile", terminal_column]].copy()
    terminal[terminal_column] = terminal[terminal_column] * 100.0

    print("Done.")
    print(f"Timeframe: {result.query['timeframe']}")
    print(f"Query state: {result.query['state']}")
    print(f"Query end: {result.query['query_end_time']}")
    print(f"Matches found: {result.query['top_k_found']} / {result.query['top_k_requested']}")
    print("Terminal return quantiles:")
    for _, row in terminal.iterrows():
        print(f"  {row['quantile']:.0%}: {row[terminal_column]:.3f}%")
    print(f"Chart: {image_path}")
    print(f"Candlestick grid: {ohlc_path}")
    print(f"Similarity diagnostics: {diagnostics_path}")
    print(f"Matches: {matches_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
