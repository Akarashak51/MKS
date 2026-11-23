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
        f"  -> Stockouts: {st['recorded_stockout_rows']:,} recorded + "
        f"{st['detected_unrecorded_stockout_rows']:,} detected unrecorded "
        f"(+{st['total_uplift_units']:,.0f} unconstrained units)"
    )

    print("\n[3/4] Workstream 3: Intermittency, Order Cycles, Macro Lead/Lag & EV Shift...")
    fd = run_feature_driver_analysis()
    print(f"  -> SBC classes: {fd['sbc_distribution']}")
    print(f"  -> Order cycles (1=Monthly, 2=Bi-Mo, 3=Quarterly): {fd['cycle_distribution']}")
    print(
        f"  -> EV share: {fd['ev_shift']['ev_2019_pct']}% (2019) -> {fd['ev_shift']['ev_2026_pct']}% (2026) "
        f"| corr with EV_Adoption_Index = {fd['ev_shift']['corr_with_ev_index']}"
    )

    print("\n[4/4] Workstream 4: Rolling-Origin 19M Backtest, Forecasting & Reconciliation...")
    fc = run_forecasting()
    for m_name, vals in fc["model_comparison"].items():
        print(f"  -> {m_name:35s} | WAPE: {vals['WAPE_Pct']:5.2f}% | Bias: {vals['Bias_Pct']:+6.2f}%")
    hl = fc["forecast_headline"]
    print(
        f"  -> Submission written: {hl['total_rows']:,} rows | P50={hl['total_p50_units']:,} units "
        f"(${hl['total_p50_revenue_usd']:,.0f}) | P90={hl['total_p90_units']:,} units | "
        f"Hierarchy discrepancy={fc['reconciliation']['max_hierarchy_discrepancy']}"
    )

    print(f"\nPipeline completed in {time.time() - t0:.1f}s.")
    print("Launch the dashboard with: python3 dashboard/app.py")


if __name__ == "__main__":
    main()
