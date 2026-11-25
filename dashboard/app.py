#!/usr/bin/env python3
"""
Celeris Automobili Industrial — Planning Control Tower Local Server.
Reads dynamically from pipeline outputs in ../outputs/ (zero hardcoded numbers).

Usage:
    python3 dashboard/app.py [--port 8050]
"""
import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

DASH_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DASH_DIR.parent
OUT_DIR = PROJECT_DIR / "outputs"
RAW_DIR = PROJECT_DIR / "data" / "raw" / "student_data"


class DataStore:
    """Loads all pipeline outputs once in memory for sub-50ms interactive filtering."""

    def __init__(self, out_dir: Path = OUT_DIR, raw_dir: Path = RAW_DIR):
        self.out_dir = out_dir
        self.raw_dir = raw_dir
        self.train = pd.read_csv(out_dir / "train_enriched.csv.gz", low_memory=False)
        self.forecast = pd.read_csv(out_dir / "forecast_full.csv.gz", low_memory=False)
        self.anomalies = pd.read_csv(out_dir / "anomaly_register.csv")
        self.interm = pd.read_csv(out_dir / "intermittency_profile.csv")
        self.macro = pd.read_csv(out_dir / "macro_clean.csv")
        sm_path = out_dir / "sku_master_clean.csv"
        ev_path = out_dir / "events_calendar_clean.csv"
        self.sku_master = pd.read_csv(sm_path if sm_path.exists() else raw_dir / "sku_master.csv")
        self.events = pd.read_csv(ev_path if ev_path.exists() else raw_dir / "events_calendar.csv")

        with open(out_dir / "data_quality_report.json") as f:
            self.dq_report = json.load(f)
        with open(out_dir / "anomaly_summary.json") as f:
            self.anomaly_summary = json.load(f)
        with open(out_dir / "backtest_metrics.json") as f:
            self.backtest = json.load(f)
        with open(out_dir / "macro_lead_lag.json") as f:
            self.macro_drivers = json.load(f)
        with open(out_dir / "agg_intervals.json") as f:
            self.agg_intervals = json.load(f)

        # Ensure Revenue_Unconstrained exists on train
        self.train["Revenue_Unconstrained"] = (
            self.train["Unconstrained_Demand"].fillna(0.0)
            * self.train["Unit_Price_USD"].fillna(0.0)
        )
        # Precompute filter lists
        self.meta = {
            "families": sorted(self.train["Product_Family"].dropna().unique().tolist()),
            "regions": sorted(self.train["Region"].dropna().unique().tolist()),
            "channels": sorted(self.train["Sales_Channel"].dropna().unique().tolist()),
            "customers": sorted(self.train["Customer_Name"].dropna().unique().tolist()),
            "skus": sorted(
                list(
                    set(self.train["SKU_ID"].dropna().unique()).union(
                        set(self.forecast["SKU_ID"].dropna().unique())
                    )
                )
            ),
            "train_range": [str(self.train["Date"].min()), str(self.train["Date"].max())],
            "forecast_range": [str(self.forecast["Date"].min()), str(self.forecast["Date"].max())],
        }

    def _apply_filters(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        out = df
        for qk, col in [
            ("family", "Product_Family"),
            ("sku", "SKU_ID"),
            ("region", "Region"),
            ("channel", "Sales_Channel"),
            ("customer", "Customer_Name"),
        ]:
            val = params.get(qk, ["ALL"])[0]
            if val and val != "ALL" and col in out.columns:
                out = out[out[col] == val]
        return out

    def _select_interval_level(self, params: dict) -> str:
        sku = params.get("sku", ["ALL"])[0]
        fam = params.get("family", ["ALL"])[0]
        reg = params.get("region", ["ALL"])[0]
        if sku and sku != "ALL":
            return "sku"
        if fam != "ALL" and reg != "ALL":
            return "family_region"
        if fam != "ALL":
            return "family"
        if reg != "ALL":
            return "region"
        return "global"

    def _apply_agg_intervals(self, f_m: pd.DataFrame, level: str) -> pd.DataFrame:
        """Apply calibrated aggregate quantiles for non-SKU views instead of summing SKU quantiles."""
        if level == "sku" or len(f_m) == 0:
            return f_m
        q_map = self.agg_intervals[level]
        p10_list, p90_list, rev90_list = [], [], []
        for r in f_m.itertuples(index=False):
            hb = str(r.H_Bucket)
            q = q_map.get(hb, {"q10": -0.15, "q90": 0.20})
            p10_list.append(max(0, int(round(r.p50 * (1.0 + q["q10"])))))
            p90_list.append(max(int(r.p50), int(round(r.p50 * (1.0 + q["q90"])))))
            rev90_list.append(round(float(r.rev_p50) * (1.0 + q["q90"]), 2))
        f_m = f_m.copy()
        f_m["p10"] = p10_list
        f_m["p90"] = p90_list
        f_m["rev_p90"] = rev90_list
        return f_m

    def get_overview(self, params: dict) -> dict:
        tr = self._apply_filters(self.train, params)
        fc = self._apply_filters(self.forecast, params)
        level = self._select_interval_level(params)

        h_m = (
            tr.groupby("Date", as_index=False)
            .agg(
                shipped=("Units_Sold_Clean", "sum"),
                unconstrained=("Unconstrained_Demand", "sum"),
                revenue=("Revenue_Unconstrained", "sum"),
            )
            .sort_values("Date")
        )
        f_m = (
            fc.groupby(["Date", "H_Bucket"], as_index=False)
            .agg(
                p10=("P10", "sum"),
                p50=("P50", "sum"),
                p90=("P90", "sum"),
                rev_p50=("Revenue_P50", "sum"),
                rev_p90=("Revenue_P90", "sum"),
            )
            .sort_values("Date")
        )
        f_m = self._apply_agg_intervals(f_m, level)

        timeline = []
        for r in h_m.itertuples(index=False):
            timeline.append({
                "date": r.Date,
                "phase": "history",
                "shipped": round(float(r.shipped), 1),
                "unconstrained": round(float(r.unconstrained), 1),
                "revenue": round(float(r.revenue), 2),
                "p10": None,
                "p50": None,
                "p90": None,
                "rev_p50": None,
            })
        for r in f_m.itertuples(index=False):
            timeline.append({
                "date": r.Date,
                "phase": "forecast",
                "shipped": None,
                "unconstrained": None,
                "revenue": None,
                "p10": int(r.p10),
                "p50": int(r.p50),
                "p90": int(r.p90),
                "rev_p50": round(float(r.rev_p50), 2),
            })

        def mix_table(df_in, col, val_col):
            if len(df_in) == 0:
                return []
            g = df_in.groupby(col)[val_col].sum().sort_values(ascending=False)
            tot = float(g.sum()) or 1.0
            return [
                {"name": k, "value": round(float(v), 1), "share_pct": round(100.0 * float(v) / tot, 1)}
                for k, v in g.items()
            ]

        return {
            "kpis": {
                "hist_total_units": round(float(tr["Unconstrained_Demand"].sum()), 0),
                "hist_shipped_units": round(float(tr["Units_Sold_Clean"].sum()), 0),
                "hist_total_revenue": round(float(tr["Revenue_Unconstrained"].sum()), 2),
                "fc_p50_units": int(f_m["p50"].sum()) if len(f_m) else 0,
                "fc_p10_units": int(f_m["p10"].sum()) if len(f_m) else 0,
                "fc_p90_units": int(f_m["p90"].sum()) if len(f_m) else 0,
                "fc_p50_revenue": round(float(f_m["rev_p50"].sum()), 2) if len(f_m) else 0.0,
                "fc_p90_revenue": round(float(f_m["rev_p90"].sum()), 2) if len(f_m) else 0.0,
                "interval_level": level,
                "active_skus": int(fc["SKU_ID"].nunique()),
                "backtest_wape": self.backtest["model_comparison"]["Reconciled & Rounded Ensemble (Final Submission)"]["WAPE_Pct"],
                "backtest_bias": self.backtest["model_comparison"]["Reconciled & Rounded Ensemble (Final Submission)"]["Bias_Pct"],
            },
            "timeline": timeline,
            "mix_family": mix_table(fc, "Product_Family", "P50"),
            "mix_region": mix_table(fc, "Region", "P50"),
            "mix_channel": mix_table(fc, "Sales_Channel", "P50"),
        }

    def get_demand_explorer(self, params: dict) -> dict:
        tr = self._apply_filters(self.train, params)
        ip = self._apply_filters(self.interm, params)

        seas = (
            tr.groupby("Month", as_index=False)["Unconstrained_Demand"]
            .mean()
            .sort_values("Month")
        )
        m_mean = float(seas["Unconstrained_Demand"].mean()) or 1.0
        seasonality = [
            {
                "month": int(r.Month),
                "avg_units": round(float(r.Unconstrained_Demand), 2),
                "index": round(float(r.Unconstrained_Demand) / m_mean, 3),
            }
            for r in seas.itertuples(index=False)
        ]

        # Drill-down table (top 40 SKU-Region-Customer combinations by recent demand)
        ip_m = ip.merge(
            self.sku_master[["SKU_ID", "Powertrain_Application", "Launch_Date", "EOL_Date", "Lifecycle_Status"]],
            on="SKU_ID",
            how="left",
        )
        top_rows = (
            ip_m.sort_values("Recent_24m_Mean", ascending=False)
            .head(50)
            .fillna("")
            .to_dict(orient="records")
        )

        sbc_counts = ip["SBC_Class"].value_counts().to_dict()
        cyc_counts = {str(k): int(v) for k, v in ip["Detected_Cycle_Period"].value_counts().items()}
        zero_pct = round(float((tr["Units_Sold_Clean"] == 0).mean() * 100.0), 1) if len(tr) else 0.0

        return {
            "global_audit": self.anomaly_summary["stockout_summary"],
            "summary": {
                "series_count": len(ip),
                "zero_month_pct": zero_pct,
                "sbc_counts": sbc_counts,
                "cycle_counts": cyc_counts,
            },
            "seasonality": seasonality,
            "drilldown": top_rows,
        }

    def get_anomalies(self, params: dict) -> dict:
        tr = self._apply_filters(self.train, params)
        an = self._apply_filters(self.anomalies, params)
        cat_filter = params.get("category", ["ALL"])[0]
        if cat_filter and cat_filter != "ALL":
            an = an[an["Category"] == cat_filter]

        # Monthly series with anomaly counts overlaid
        m_ts = (
            tr.groupby("Date", as_index=False)
            .agg(shipped=("Units_Sold_Clean", "sum"), unconstrained=("Unconstrained_Demand", "sum"))
            .sort_values("Date")
        )
        an_m = an.groupby("Date").size().to_dict()
        series = [
            {
                "date": r.Date,
                "shipped": round(float(r.shipped), 1),
                "unconstrained": round(float(r.unconstrained), 1),
                "anomaly_count": int(an_m.get(r.Date, 0)),
            }
            for r in m_ts.itertuples(index=False)
        ]

        return {
            "summary": self.anomaly_summary,
            "filtered_count": len(an),
            "filtered_by_category": an["Category"].value_counts().to_dict(),
            "filtered_by_type": an["Anomaly_Type"].value_counts().head(15).to_dict(),
            "series": series,
            "rows": an.head(150).fillna("").to_dict(orient="records"),
        }

    def get_what_if(self, params: dict) -> dict:
        fc = self._apply_filters(self.forecast, params).copy()
        pmi_delta = float(params.get("pmi_delta", ["0"])[0])
        auto_delta = float(params.get("auto_delta", ["0"])[0])
        ev_delta = float(params.get("ev_delta", ["0"])[0])
        price_delta = float(params.get("price_delta", ["0"])[0])

        # Model-derived elasticities extracted directly from fitted Ridge regression weights w_ml
        el = self.backtest["what_if_elasticities"]
        sm_pt = self.sku_master.set_index("SKU_ID")["Powertrain_Application"].to_dict()
        pt = fc["SKU_ID"].map(sm_pt).fillna("Universal")

        mult = np.ones(len(fc))
        mult *= np.clip(np.exp(el["pmi_pct_per_pt"] * pmi_delta), 0.65, 1.40)
        mult *= np.clip(np.exp(el["auto_yoy_pct_per_pct"] * auto_delta), 0.70, 1.35)
        ev_slope = np.where(
            pt == "EV",
            el["ev_sku_pct_per_pt"],
            np.where(pt == "ICE", el["ice_sku_pct_per_pt"], el["ev_universal_pct_per_pt"]),
        )
        mult *= np.clip(np.exp(ev_slope * ev_delta), 0.65, 1.50)
        mult *= np.clip(np.exp(el["price_pct_per_1pct"] * price_delta), 0.70, 1.30)

        fc["Scen_P50"] = np.round(fc["P50"] * mult).astype(int)
        fc["Scen_P90"] = np.round(fc["P90"] * mult).astype(int)
        fc["Scen_Price"] = fc["Unit_Price_Projected"] * (1.0 + price_delta / 100.0)
        fc["Scen_Rev"] = np.round(fc["Scen_P50"] * fc["Scen_Price"], 2)

        monthly = (
            fc.groupby("Date", as_index=False)
            .agg(
                base_p50=("P50", "sum"),
                scen_p50=("Scen_P50", "sum"),
                base_rev=("Revenue_P50", "sum"),
                scen_rev=("Scen_Rev", "sum"),
            )
            .sort_values("Date")
        )
        base_u = int(fc["P50"].sum())
        scen_u = int(fc["Scen_P50"].sum())
        base_r = float(fc["Revenue_P50"].sum())
        scen_r = float(fc["Scen_Rev"].sum())

        return {
            "elasticities": el,
            "deltas": {
                "pmi_delta": pmi_delta,
                "auto_delta": auto_delta,
                "ev_delta": ev_delta,
                "price_delta": price_delta,
            },
            "base_units": base_u,
            "scenario_units": scen_u,
            "unit_diff": scen_u - base_u,
            "unit_diff_pct": round(100.0 * (scen_u - base_u) / max(1, base_u), 2),
            "base_revenue": round(base_r, 2),
            "scenario_revenue": round(scen_r, 2),
            "rev_diff": round(scen_r - base_r, 2),
            "monthly": monthly.to_dict(orient="records"),
        }

    def get_procurement(self, params: dict) -> dict:
        fc = self._apply_filters(self.forecast, params)
        level = self._select_interval_level(params)
        monthly = (
            fc.groupby(["Date", "H_Bucket"], as_index=False)
            .agg(
                p10=("P10", "sum"),
                p50=("P50", "sum"),
                p90=("P90", "sum"),
                rev_p50=("Revenue_P50", "sum"),
                rev_p90=("Revenue_P90", "sum"),
            )
            .sort_values("Date")
        )
        monthly = self._apply_agg_intervals(monthly, level)
        monthly["safety_buffer"] = monthly["p90"] - monthly["p50"]

        by_sku = (
            fc.groupby(["SKU_ID", "Product_Family", "Region"], as_index=False)
            .agg(
                p50_19m=("P50", "sum"),
                p90_19m=("P90", "sum"),
                budget_p90=("Revenue_P90", "sum"),
            )
        )
        by_sku["safety_buffer"] = by_sku["p90_19m"] - by_sku["p50_19m"]
        by_sku = by_sku.sort_values("budget_p90", ascending=False).head(40)

        tot_p50 = int(monthly["p50"].sum()) if len(monthly) else 0
        tot_p90 = int(monthly["p90"].sum()) if len(monthly) else 0
        tot_rev90 = round(float(monthly["rev_p90"].sum()), 2) if len(monthly) else 0.0
        return {
            "interval_level": level,
            "total_p50": tot_p50,
            "total_p90": tot_p90,
            "total_safety_buffer": tot_p90 - tot_p50,
            "total_budget_p90": tot_rev90,
            "monthly": monthly.to_dict(orient="records"),
            "top_skus": by_sku.to_dict(orient="records"),
        }

    def get_lifecycle(self, params: dict) -> dict:
        eol_skus = self.sku_master[
            self.sku_master["Lifecycle_Status"].isin(["EOL Announced", "Planned Launch", "Discontinued"])
        ].copy()

        # Transition pairs active in forecast window: (PMP-003->PMP-020, PTC-005->PTC-029, THM-001->THM-012)
        pairs = []
        planned = self.sku_master[self.sku_master["Lifecycle_Status"] == "Planned Launch"]
        for r in planned.itertuples(index=False):
            old_sku = r.Replaces_SKU
            new_sku = r.SKU_ID
            h_old = self.train[self.train["SKU_ID"] == old_sku].groupby("Date")["Unconstrained_Demand"].sum()
            f_old = self.forecast[self.forecast["SKU_ID"] == old_sku].groupby("Date")["P50"].sum()
            f_new = self.forecast[self.forecast["SKU_ID"] == new_sku].groupby("Date")["P50"].sum()

            all_dts = sorted(set(h_old.index[-18:]).union(set(f_old.index)).union(set(f_new.index)))
            tl = []
            for dt in all_dts:
                tl.append({
                    "date": dt,
                    "phase": "history" if dt <= "2026-10" else "forecast",
                    "old_units": round(float(h_old.get(dt, f_old.get(dt, 0.0))), 1),
                    "new_units": round(float(f_new.get(dt, 0.0)), 1),
                })
            old_eol = self.sku_master.loc[self.sku_master["SKU_ID"] == old_sku, "EOL_Date"].iloc[0]
            pairs.append({
                "old_sku": old_sku,
                "new_sku": new_sku,
                "family": r.Product_Family,
                "old_eol": str(old_eol),
                "new_launch": str(r.Launch_Date),
                "timeline": tl,
            })

        return {
            "ramp_up_curve": self.backtest["calibrated_ramp_up_curve"],
            "ramp_down_curve": self.backtest["calibrated_ramp_down_curve"],
            "active_transitions": pairs,
            "lifecycle_table": eol_skus.fillna("").to_dict(orient="records"),
        }

    def get_stockouts(self, params: dict) -> dict:
        tr = self._apply_filters(self.train, params)
        st = tr[tr["Stockout_Status"] != "none"]

        monthly = (
            tr.groupby("Date", as_index=False)
            .agg(
                shipped=("Units_Sold_Clean", "sum"),
                unconstrained=("Unconstrained_Demand", "sum"),
                lost_est=("Estimated_Lost_Units", "sum"),
            )
            .sort_values("Date")
        )
        by_fam = (
            st.groupby(["Product_Family", "Stockout_Status"], as_index=False)["Estimated_Lost_Units"]
            .sum()
            .pivot(index="Product_Family", columns="Stockout_Status", values="Estimated_Lost_Units")
            .fillna(0.0)
            .reset_index()
        )
        by_reason = (
            st[st["Stockout_Status"] == "recorded"]
            .groupby("Reason", as_index=False)["Lost_Units"]
            .sum()
            .sort_values("Lost_Units", ascending=False)
        )
        return {
            "summary": {
                "recorded_rows": int((st["Stockout_Status"] == "recorded").sum()),
                "detected_rows": int((st["Stockout_Status"] == "detected_unrecorded").sum()),
                "lost_units_recorded": round(float(st.loc[st["Stockout_Status"] == "recorded", "Estimated_Lost_Units"].sum()), 0),
                "lost_units_detected": round(float(st.loc[st["Stockout_Status"] == "detected_unrecorded", "Estimated_Lost_Units"].sum()), 0),
                "lost_revenue_est": round(float((st["Estimated_Lost_Units"] * st["Unit_Price_USD"].fillna(0)).sum()), 2),
            },
            "monthly": monthly.to_dict(orient="records"),
            "by_family": by_fam.to_dict(orient="records"),
            "by_reason": by_reason.to_dict(orient="records"),
        }

    def get_exceptions(self, params: dict) -> dict:
        fc = self._apply_filters(self.forecast, params)
        an = self._apply_filters(self.anomalies, params)

        exceptions = []
        # 1. Lost Accounts, Customer Switches, Cycle Changes, MDM contradictions from anomaly register
        key_types = [
            "Lost_Account",
            "Customer_Switch",
            "Ordering_Cycle_Change",
            "Sales_Before_Master_Launch_Date",
            "Permanent_Capacity_Step_Up",
        ]
        for r in an[an["Anomaly_Type"].isin(key_types)].head(80).itertuples(index=False):
            exceptions.append({
                "Priority": "P1 - High" if r.Severity == "High" else "P2 - Medium",
                "SKU_ID": r.SKU_ID,
                "Region": r.Region,
                "Product_Family": r.Product_Family,
                "Exception_Type": r.Anomaly_Type,
                "Date_Context": r.Date,
                "Detail": r.Reason,
                "Action": r.Recommended_Action,
            })

        # 2. High forecast uncertainty SKUs (P90 - P50 spread)
        spread = (
            fc.groupby(["SKU_ID", "Region", "Product_Family"], as_index=False)
            .agg(p50=("P50", "sum"), p90=("P90", "sum"), rev_p90=("Revenue_P90", "sum"))
        )
        spread["buffer_units"] = spread["p90"] - spread["p50"]
        spread["ratio"] = spread["p90"] / np.maximum(spread["p50"], 1)
        high_unc = spread[spread["p50"] >= 200].sort_values("buffer_units", ascending=False).head(25)
        for r in high_unc.itertuples(index=False):
            exceptions.append({
                "Priority": "P2 - Medium",
                "SKU_ID": r.SKU_ID,
                "Region": r.Region,
                "Product_Family": r.Product_Family,
                "Exception_Type": "High_Forecast_Uncertainty",
                "Date_Context": "2026-11..2028-05",
                "Detail": f"19m P50={int(r.p50):,} vs P90={int(r.p90):,} (+{int(r.buffer_units):,} safety units, {r.ratio:.2f}x)",
                "Action": "Review supplier lead-time & hold P90 buffer stock",
            })

        return {"total_exceptions": len(exceptions), "exceptions": exceptions}


STORE: DataStore | None = None


class ControlTowerHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200):
        import math
        def clean_val(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            if isinstance(v, dict):
                return {k: clean_val(val) for k, val in v.items()}
            if isinstance(v, list):
                return [clean_val(x) for x in v]
            return v
        body = json.dumps(clean_val(payload), default=str, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, download_name: str | None = None):
        if not path.exists():
            self.send_error(404, "File not found")
            return
        data = path.read_bytes()
        ctype, _ = mimetypes.guess_type(str(path))
        self.send_response(200)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass  # keep console clean

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._send_file(DASH_DIR / "index.html")
            return
        if path == "/api/meta":
            self._send_json(STORE.meta)
            return
        if path == "/api/overview":
            self._send_json(STORE.get_overview(params))
            return
        if path == "/api/data-health":
            self._send_json(STORE.dq_report)
            return
        if path == "/api/demand-explorer":
            self._send_json(STORE.get_demand_explorer(params))
            return
        if path == "/api/anomalies":
            self._send_json(STORE.get_anomalies(params))
            return
        if path == "/api/forecast-accuracy":
            self._send_json({
                "overview": STORE.get_overview(params),
                "backtest": STORE.backtest,
            })
            return
        if path == "/api/macro-drivers":
            self._send_json(STORE.macro_drivers)
            return
        if path == "/api/what-if":
            self._send_json(STORE.get_what_if(params))
            return
        if path == "/api/procurement":
            self._send_json(STORE.get_procurement(params))
            return
        if path == "/api/lifecycle":
            self._send_json(STORE.get_lifecycle(params))
            return
        if path == "/api/stockouts":
            self._send_json(STORE.get_stockouts(params))
            return
        if path == "/api/exceptions":
            self._send_json(STORE.get_exceptions(params))
            return
        if path.startswith("/api/download/"):
            fname = path.split("/api/download/")[-1]
            allowed = {
                "submission.csv": OUT_DIR / "submission.csv",
                "anomaly_register.csv": OUT_DIR / "anomaly_register.csv",
                "intermittency_profile.csv": OUT_DIR / "intermittency_profile.csv",
                "macro_clean.csv": OUT_DIR / "macro_clean.csv",
                "data_quality_report.json": OUT_DIR / "data_quality_report.json",
                "backtest_metrics.json": OUT_DIR / "backtest_metrics.json",
            }
            if fname in allowed:
                self._send_file(allowed[fname], download_name=fname)
                return
        self.send_error(404, "Endpoint not found")


def main():
    global STORE
    parser = argparse.ArgumentParser(description="Celeris Planning Control Tower Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    print("Loading pipeline outputs from:", OUT_DIR)
    STORE = DataStore()
    server = ThreadingHTTPServer((args.host, args.port), ControlTowerHandler)
    print(f"Celeris Control Tower running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Control Tower.")
        server.server_close()


if __name__ == "__main__":
    main()
