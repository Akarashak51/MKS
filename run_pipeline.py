#!/usr/bin/env python3
"""
One-command end-to-end pipeline for Celeris Automobili Industrial
Demand Forecasting & Planning Control Tower.

Usage:
    python3 run_pipeline.py
"""
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from clean import run_cleaning
from anomalies import detect_anomalies_and_unconstrain
from features_drivers import run_feature_driver_analysis
from forecast import run_forecasting


def main():
    t0 = time.time()
    print("=" * 72)
    print("Celeris Automobili Industrial — End-to-End Forecasting Pipeline")
    print("=" * 72)

    print("\n[1/4] Workstream 1: Data Cleaning & Quality Reconciliation...")
    dq = run_cleaning()
    ts = dq["train_summary"]
    print(
        f"  -> Cleaned train: {ts['rows_before']:,} raw -> {ts['rows_after_dedup']:,} dedup -> "
        f"{ts['rows_after_panel_completion']:,} panel rows ({ts['missing_panel_rows_added']:,} explicit NaN)"
    )
    print(
        f"  -> Reconciliation mismatches: {ts['reconciliation_mismatches_before']:,} -> "
        f"{ts['reconciliation_mismatches_after']:,}"
    )

    print("\n[2/4] Workstream 2: Anomaly Detection & Unconstrained Demand...")
    an = detect_anomalies_and_unconstrain()
    st = an["stockout_summary"]
    print(
        f"  -> Logged {an['total_register_entries']:,} register entries "
        f"({an['by_category'].get('Data Error', 0):,} data errors, "
        f"{an['total_register_entries'] - an['by_category'].get('Data Error', 0):,} real events/contradictions)"
    )
    print(
        f"  -> Backorder catch-up removed: -{st['backorder_catchup_units_subtracted']:,.0f} units "
        f"across {st['backorder_catchup_rows_adjusted']:,} post-backorder months"
    )
    print(
        f"  -> Stockouts: {st['recorded_stockout_rows']:,} recorded + "
        f"{st['detected_unrecorded_stockout_rows']:,} detected unrecorded "
        f"(recall={st['detector_validation_recall_overall_pct']}%, "
        f"general rate={st['general_unrecorded_detection_rate_pct']}%, "
        f"net uplift=+{st['total_uplift_units']:,.0f} units)"
    )

    print("\n[3/4] Workstream 3: Intermittency, Order Cycles, Macro Lead/Lag & EV Shift...")
    fd = run_feature_driver_analysis()
    print(f"  -> SBC classes: {fd['sbc_distribution']}")
    print(f"  -> Order cycles (1=Monthly, 2=Bi-Mo, 3=Quarterly): {fd['cycle_distribution']}")
    print(f"  -> COVID window excluded: {fd['covid_window_excluded']}")
    ev = fd["ev_shift"]
    print(
        f"  -> EV share: {ev['ev_2019_pct']}% (2019) -> {ev['ev_2026_pct']}% (2026) "
        f"| level corr = {ev['level_corr']}, detrended MoM corr = {ev['detrended_corr']}"
    )

    print("\n[4/4] Workstream 4: 4-Origin 19M Backtest, Forecasting & Reconciliation...")
    fc = run_forecasting()
    for m_name, vals in fc["model_comparison"].items():
        print(f"  -> {m_name:48s} | WAPE: {vals['WAPE_Pct']:5.2f}% | Bias: {vals['Bias_Pct']:+6.2f}%")
    agg = fc["aggregate_accuracy"]
    print(
        f"  -> Aggregate WAPE: Global Monthly = {agg['global_monthly_total']['reconciled_wape_pct']}% "
        f"(vs SN {agg['global_monthly_total']['snaive_wape_pct']}%) | "
        f"Family-Region Monthly = {agg['family_region_monthly']['reconciled_wape_pct']}% "
        f"(vs SN {agg['family_region_monthly']['snaive_wape_pct']}%)"
    )
    sc = fc["shipped_actuals_check"]
    print(
        f"  -> Shipped-actuals check: Reconciled WAPE = {sc['reconciled_vs_shipped_wape_pct']}%, "
        f"Bias = {sc['reconciled_vs_shipped_bias_pct']:+.2f}%"
    )
    hl = fc["forecast_headline"]
    print(
        f"  -> Submission written: {hl['total_rows']:,} rows | "
        f"P50 = {hl['total_p50_units']:,} units (${hl['total_p50_revenue_usd']:,.0f}) | "
        f"Calibrated Global P90 = {hl['total_p90_units_global_calibrated']:,} units "
        f"(SKU-summed P90 = {hl['sku_sum_p90_units_for_sku_level_buffer']:,})"
    )

    print(f"\nPipeline completed cleanly in {time.time() - t0:.1f}s.")
    print("Launch the dashboard with: python3 dashboard/app.py --port 8050")


if __name__ == "__main__":
    main()
