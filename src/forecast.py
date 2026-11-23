import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUT_DIR, RAW_DIR
from features_drivers import detect_cycle_and_phase


def month_to_int(ym: str) -> int:
    y, m = ym.split("-")
    return int(y) * 12 + int(m)


def build_holiday_flags(dates: pd.Series, regions: pd.Series, events: pd.DataFrame) -> pd.DataFrame:
    """Create binary event/holiday indicators aligned to (Date, Region)."""
    # Extract LNY and Ramadan months from events_calendar.csv
    lny_months = set()
    ramadan_months = set()
    for r in events.itertuples(index=False):
        desc = str(r.Description).lower()
        sd = str(r.Start_Date)[:7]
        ed = str(r.End_Date)[:7] if pd.notna(r.End_Date) else sd
        if "lunar new year" in desc:
            lny_months.add(sd)
            lny_months.add(ed)
        elif "ramadan" in desc:
            ramadan_months.add(sd)
            ramadan_months.add(ed)

    m_num = dates.str[5:7].astype(int)
    out = pd.DataFrame(index=dates.index)
    out["is_NA_July"] = ((regions == "North America") & (m_num == 7)).astype(float)
    out["is_EU_Aug"] = ((regions == "Europe") & (m_num == 8)).astype(float)
    out["is_APAC_LNY"] = ((regions == "Asia Pacific") & (dates.isin(lny_months))).astype(float)
    out["is_MEA_Ramadan"] = ((regions == "Middle East & Africa") & (dates.isin(ramadan_months))).astype(float)
    return out


def build_price_hike_map(events: pd.DataFrame) -> dict[str, float]:
    """Map YYYY-MM of Jan price increase -> pct increase (e.g. '2023-01': 9.0)."""
    import re
    hike_map = {}
    for r in events.itertuples(index=False):
        if r.Event_Type == "Price Change" and "1 January" in str(r.Description):
            m = re.search(r"(\d+\.?\d*)%", str(r.Description))
            if m:
                hike_map[str(r.Start_Date)[:7]] = float(m.group(1))
    return hike_map


def calibrate_replacement_curves(df: pd.DataFrame, sku_master: pd.DataFrame) -> tuple[dict, dict]:
    """Learn ramp-up curve for new launches and ramp-down curve for EOL SKUs from historical pairs."""
    rep_pairs = sku_master[sku_master["Replaces_SKU"].notna() & (sku_master["Lifecycle_Status"] == "Active")]
    ramp_up_samples = {m: [] for m in range(6)}
    ramp_down_samples = {m: [] for m in range(4)}  # months before EOL: 0=EOL month, 1=1m before, etc.

    for r in rep_pairs.itertuples(index=False):
        new_sku = r.SKU_ID
        old_sku = r.Replaces_SKU
        launch_t = month_to_int(r.Launch_Date)
        old_row = sku_master[sku_master["SKU_ID"] == old_sku]
        if len(old_row) == 0:
            continue
        eol_dt = old_row["EOL_Date"].iloc[0]
        eol_t = month_to_int(eol_dt) if pd.notna(eol_dt) else None

        old_df = df[df["SKU_ID"] == old_sku].groupby("Date")["Unconstrained_Demand"].sum()
        new_df = df[df["SKU_ID"] == new_sku].groupby("Date")["Unconstrained_Demand"].sum()
        if len(old_df) < 12 or len(new_df) < 6:
            continue

        # Baseline of old SKU 6-18 months before new launch
        old_base_vals = [
            v for dt, v in old_df.items()
            if (launch_t - 18) <= month_to_int(dt) <= (launch_t - 4) and v > 0
        ]
        if not old_base_vals:
            continue
        old_base = float(np.median(old_base_vals))
        if old_base <= 0:
            continue

        for dt, v in new_df.items():
            age = month_to_int(dt) - launch_t
            if 0 <= age < 6:
                ramp_up_samples[age].append(min(2.0, v / old_base))

        if eol_t is not None:
            for dt, v in old_df.items():
                rem = eol_t - month_to_int(dt)
                if 0 <= rem < 4:
                    ramp_down_samples[rem].append(min(1.5, v / old_base))

    ramp_up = {
        m: round(float(np.median(vals)) if vals else min(1.0, 0.35 + 0.2 * m), 3)
        for m, vals in ramp_up_samples.items()
    }
    ramp_down = {
        m: round(float(np.median(vals)) if vals else (0.35 + 0.2 * m), 3)
        for m, vals in ramp_down_samples.items()
    }
    return ramp_up, ramp_down


def tsb_forecast_series(y: np.ndarray, alpha_d: float = 0.18, alpha_p: float = 0.18) -> tuple[float, float, float]:
    """Teunter-Syntetos-Babai (TSB) intermittent demand state: returns (p_t, z_t, forecast_rate)."""
    valid = y[~np.isnan(y)]
    if len(valid) == 0:
        return 0.0, 0.0, 0.0
    pos = valid[valid > 0]
    if len(pos) == 0:
        return 0.0, 0.0, 0.0

    p = float((valid[: min(6, len(valid))] > 0).mean())
    z = float(pos[: min(4, len(pos))].mean())
    for val in valid:
        if val > 0:
            p = p + alpha_p * (1.0 - p)
            z = z + alpha_d * (val - z)
        else:
            p = p + alpha_p * (0.0 - p)
    return float(p), float(z), float(p * z)


def fit_and_predict_origin(
    train_df: pd.DataFrame,
    future_keys: pd.DataFrame,
    macro: pd.DataFrame,
    sku_master: pd.DataFrame,
    events: pd.DataFrame,
    origin_date: str,
    ramp_up_curve: dict,
    ramp_down_curve: dict,
) -> pd.DataFrame:
    """
    Leak-free forecast from origin_date (using strictly train_df['Date'] <= origin_date)
    for all rows in future_keys (Date > origin_date).
    Produces: pred_snaive, pred_ma6, pred_tsb_cycle, pred_global_ml, pred_ensemble.
    """
    hist = train_df[train_df["Date"] <= origin_date].copy()
    origin_t = month_to_int(origin_date)
    hist["t_int"] = hist["Date"].map(month_to_int)
    hist["y"] = hist["Unconstrained_Demand_For_Model"].fillna(0.0).clip(lower=0.0)

    # Merge Powertrain_Application, Launch_Date, EOL_Date, Replaces_SKU
    sm = sku_master[["SKU_ID", "Powertrain_Application", "Launch_Date", "EOL_Date", "Replaces_SKU", "Lifecycle_Status"]].copy()
    hist = hist.merge(sm, on="SKU_ID", how="left")

    # 1. Compute regional-family seasonal indices (from clean non-COVID months)
    clean_seas = hist[~hist["Is_Covid_Disruption"] & (hist["y"] > 0)]
    rf_month_mean = clean_seas.groupby(["Product_Family", "Region", "Month"])["y"].mean()
    rf_overall_mean = clean_seas.groupby(["Product_Family", "Region"])["y"].mean()
    rf_seas_idx = (rf_month_mean / (rf_overall_mean + 1e-6)).clip(0.55, 1.55).to_dict()

    # Family-level recent YoY growth (EV vs ICE vs Universal) for subtle structural drift
    last_12_start = origin_t - 11
    prev_12_start = origin_t - 23
    pt_last12 = hist[hist["t_int"].between(last_12_start, origin_t)].groupby(["Product_Family", "Powertrain_Application"])["y"].mean()
    pt_prev12 = hist[hist["t_int"].between(prev_12_start, last_12_start - 1)].groupby(["Product_Family", "Powertrain_Application"])["y"].mean()
    pt_growth_annual = ((pt_last12 / (pt_prev12 + 1e-6)) - 1.0).clip(-0.18, 0.18).to_dict()

    # 2. Compute per-(SKU_ID, Region) state strictly at origin_date
    series_state = {}
    for (sid, reg), g in hist.groupby(["SKU_ID", "Region"], sort=False):
        g = g.sort_values("t_int")
        y_vals = g["y"].values
        t_vals = g["t_int"].values
        m_vals = g["Month"].values

        # Recent slices
        last24 = g[g["t_int"] > origin_t - 24]
        last12 = g[g["t_int"] > origin_t - 12]
        last6 = g[g["t_int"] > origin_t - 6]
        last3 = g[g["t_int"] > origin_t - 3]

        y24 = last24["y"].values
        t24 = last24["t_int"].values
        y12 = last12["y"].values
        y6 = last6["y"].values
        y3 = last3["y"].values

        # Robust recent averages (dampen extreme 1-off spikes > 2.5x median)
        pos12 = y12[y12 > 0]
        pos24 = y24[y24 > 0]
        med_pos = float(np.median(pos24)) if len(pos24) > 0 else 0.0
        y12_wins = np.minimum(y12, 2.5 * med_pos) if med_pos > 0 else y12
        y6_wins = np.minimum(y6, 2.5 * med_pos) if med_pos > 0 else y6

        ma6 = float(y6.mean()) if len(y6) > 0 else 0.0
        ma12 = float(y12_wins.mean()) if len(y12_wins) > 0 else ma6
        ma3 = float(y3.mean()) if len(y3) > 0 else ma6
        nz24 = float((y24 > 0).mean()) if len(y24) > 0 else 0.0
        nz12 = float((y12 > 0).mean()) if len(y12) > 0 else 0.0

        # SBC classification on last 24m
        if len(pos24) == 0:
            adi, cv2, sbc = 99.0, 0.0, "Inactive/Zero"
        else:
            adi = float(len(y24) / len(pos24))
            mp = float(pos24.mean())
            sp = float(pos24.std(ddof=0)) if len(pos24) > 1 else 0.0
            cv2 = float((sp / mp) ** 2) if mp > 0 else 0.0
            if adi < 1.32 and cv2 < 0.49:
                sbc = "Smooth"
            elif adi < 1.32:
                sbc = "Erratic"
            elif cv2 < 0.49:
                sbc = "Intermittent"
            else:
                sbc = "Lumpy"

        # Lost account check at origin_date
        is_lost = bool((y_vals > 0).sum() >= 6 and len(y_vals) >= 12 and np.all(y_vals[-6:] == 0))

        # Cycle detection on last 24m
        period, phase, cyc_str = detect_cycle_and_phase(y24, t24)

        # TSB state on last 36m (so structural shifts/customer switches adapt quickly)
        p_tsb, z_tsb, rate_tsb = tsb_forecast_series(y_vals[-36:])

        # Seasonal Naive map (last 12m exact month values)
        snaive_map = {}
        for mm, yy in zip(last12["Month"].values, last12["y"].values):
            snaive_map[int(mm)] = float(yy)

        # Series-specific seasonality shrunk toward regional-family seasonal index
        fam = g["Product_Family"].iloc[0]
        pt = g["Powertrain_Application"].iloc[0]
        own_seas = {}
        if len(pos24) >= 12:
            overall_pos = float(pos24.mean())
            for mm in range(1, 13):
                vals_m = last24.loc[(last24["Month"] == mm) & (last24["y"] > 0), "y"].values
                rf_s = rf_seas_idx.get((fam, reg, mm), 1.0)
                if len(vals_m) > 0 and overall_pos > 0:
                    w = min(0.55, 0.25 * len(vals_m))
                    own_seas[mm] = float(np.clip(w * (vals_m.mean() / overall_pos) + (1.0 - w) * rf_s, 0.45, 1.75))
                else:
                    own_seas[mm] = float(rf_s)
        else:
            for mm in range(1, 13):
                own_seas[mm] = float(rf_seas_idx.get((fam, reg, mm), 1.0))

        # Recent Helios India step-up detection (if post-2025-09 exists in hist)
        cust_latest = g["Customer_Name"].iloc[-1]
        if cust_latest == "Helios Auto Parts" and reg == "India" and origin_date >= "2025-11":
            post_h = g[g["Date"] >= "2025-09"]["y"].values
            if len(post_h) >= 3:
                ma12 = float(post_h.mean())
                z_tsb = float(post_h[post_h > 0].mean()) if (post_h > 0).any() else z_tsb

        # Customer switch recency: if customer switched within last 18m, use post-switch mean
        c_arr = g["Customer_Name"].values
        sw = np.where(c_arr[1:] != c_arr[:-1])[0]
        if len(sw) > 0:
            last_sw_idx = sw[-1] + 1
            if len(g) - last_sw_idx <= 18 and len(g) - last_sw_idx >= 3:
                post_sw = y_vals[last_sw_idx:]
                ma12 = float(post_sw.mean())
                p_tsb, z_tsb, rate_tsb = tsb_forecast_series(post_sw)

        # Level trend (recent 6m vs prior 6m, heavily damped)
        if len(y12) >= 12 and y12[:6].mean() > 3:
            raw_tr = (y12[6:].mean() / (y12[:6].mean() + 1e-6)) - 1.0
            trend_6m = float(np.clip(raw_tr, -0.25, 0.25))
        else:
            trend_6m = 0.0

        series_state[(sid, reg)] = {
            "ma6": ma6,
            "ma12": ma12,
            "ma3": ma3,
            "p_tsb": p_tsb,
            "z_tsb": z_tsb,
            "rate_tsb": rate_tsb,
            "nz12": nz12,
            "nz24": nz24,
            "adi": adi,
            "cv2": cv2,
            "sbc": sbc,
            "is_lost": is_lost,
            "period": period,
            "phase": phase,
            "cyc_str": cyc_str,
            "snaive_map": snaive_map,
            "own_seas": own_seas,
            "trend_6m": trend_6m,
            "fam": fam,
            "pt": pt,
            "pt_growth": float(pt_growth_annual.get((fam, pt), 0.0)),
            "last_price": float(g["Unit_Price_USD"].dropna().iloc[-1]) if g["Unit_Price_USD"].notna().any() else 500.0,
        }

    # 3. Fit Global ML Model (L2-Regularized Ridge Regression on log1p(y) with macro + event + series + interaction features)
    # Prepare macro with 2-month structural lead lag
    m_df = macro.sort_values("Date").copy()
    macro_cols = [
        "ISM_Manufacturing_PMI",
        "Auto_Production_YoY_Pct",
        "Vehicle_Inventory_Days",
        "New_Vehicle_Orders_Index",
        "Industrial_Util_Rate_Pct",
        "EV_Adoption_Index",
    ]
    for c in macro_cols:
        m_df[f"{c}_lag2"] = m_df[c].shift(2).bfill()
    lag2_cols = [f"{c}_lag2" for c in macro_cols]
    # Standardize macro features using strictly history <= origin_date
    m_hist_stats = m_df[m_df["Date"] <= origin_date][lag2_cols]
    m_mean = m_hist_stats.mean()
    m_std = m_hist_stats.std().replace(0, 1.0)
    for c in lag2_cols:
        m_df[f"{c}_z"] = (m_df[c] - m_mean[c]) / m_std[c]
    z_macro_cols = [f"{c}_z" for c in lag2_cols]

    hike_map = build_price_hike_map(events)

    def make_ml_features(frame: pd.DataFrame, is_train: bool) -> np.ndarray:
        f = frame.merge(m_df[["Date"] + z_macro_cols], on="Date", how="left")
        hol = build_holiday_flags(f["Date"], f["Region"], events)
        mm = f["Month"].astype(float).values
        sin_m = np.sin(2.0 * np.pi * mm / 12.0)
        cos_m = np.cos(2.0 * np.pi * mm / 12.0)

        # Next month price hike (Dec forward-buying) and current month Jan price hike
        next_ym = [
            f"{int(d[:4]) + (1 if int(d[5:7]) == 12 else 0):04d}-{(int(d[5:7]) % 12) + 1:02d}"
            for d in f["Date"].values
        ]
        pre_hike = np.array([hike_map.get(ny, 0.0) / 10.0 for ny in next_ym])
        post_hike = np.array([hike_map.get(d, 0.0) / 10.0 for d in f["Date"].values])

        # Series baseline features (log1p scale)
        base_level = np.log1p(f["feat_base_level"].fillna(0.0).clip(lower=0.0).values)
        seas_log = np.log(f["feat_seas_idx"].fillna(1.0).clip(0.4, 2.0).values)
        on_cyc = f["feat_on_cycle"].fillna(1.0).values
        nz_rate = f["feat_nz_rate"].fillna(1.0).values

        # Powertrain EV/ICE interactions with EV_Adoption_Index_lag2_z
        is_ev = (f["Powertrain_Application"] == "EV").astype(float).values
        is_ice = (f["Powertrain_Application"] == "ICE").astype(float).values
        ev_z = f["EV_Adoption_Index_lag2_z"].fillna(0.0).values
        pmi_z = f["ISM_Manufacturing_PMI_lag2_z"].fillna(0.0).values
        auto_z = f["Auto_Production_YoY_Pct_lag2_z"].fillna(0.0).values
        ord_z = f["New_Vehicle_Orders_Index_lag2_z"].fillna(0.0).values
        inv_z = f["Vehicle_Inventory_Days_lag2_z"].fillna(0.0).values
        util_z = f["Industrial_Util_Rate_Pct_lag2_z"].fillna(0.0).values

        # Disruption / structural features
        if is_train:
            is_cov = f["Is_Covid_Disruption"].astype(float).values
            is_nex = f["Is_Nexon_Strike"].astype(float).values
            is_red = f["Is_Red_Sea"].astype(float).values
        else:
            is_cov = np.zeros(len(f))
            is_nex = np.zeros(len(f))
            is_red = np.zeros(len(f))

        # Family indicators interacted with PMI for family-specific macro sensitivity
        fam_pmi = []
        for fam_name in [
            "Brake & Safety Systems", "Chassis & Structural", "Electrical Assemblies",
            "Hydraulic Systems", "Powertrain Components", "Precision Machined Parts",
        ]:
            fm = (f["Product_Family"] == fam_name).astype(float).values
            fam_pmi.append(fm)
            fam_pmi.append(fm * pmi_z)

        cols = [
            np.ones(len(f)),
            base_level,
            seas_log,
            on_cyc,
            nz_rate,
            base_level * on_cyc,
            sin_m,
            cos_m,
            hol["is_NA_July"].values,
            hol["is_EU_Aug"].values,
            hol["is_APAC_LNY"].values,
            hol["is_MEA_Ramadan"].values,
            pre_hike,
            post_hike,
            pmi_z,
            auto_z,
            ord_z,
            inv_z,
            util_z,
            ev_z,
            is_ev,
            is_ice,
            is_ev * ev_z,
            is_ice * ev_z,
            is_cov,
            is_nex,
            is_red,
        ] + fam_pmi
        return np.column_stack(cols)

    # Build leak-free training rows from last 36 months of history using trailing rolling stats
    ml_hist = hist[hist["t_int"] > origin_t - 42].copy().sort_values(["SKU_ID", "Region", "t_int"])
    grp_ml = ml_hist.groupby(["SKU_ID", "Region"], sort=False)
    ml_hist["feat_base_level"] = grp_ml["y"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=3).mean()
    )
    ml_hist["feat_nz_rate"] = grp_ml["y"].transform(
        lambda s: (s.shift(1) > 0).astype(float).rolling(12, min_periods=3).mean()
    )
    ml_hist["feat_seas_idx"] = [
        rf_seas_idx.get((fam, reg, int(m)), 1.0)
        for fam, reg, m in zip(ml_hist["Product_Family"], ml_hist["Region"], ml_hist["Month"])
    ]
    # On-cycle indicator using series_state period/phase
    ml_hist["feat_on_cycle"] = [
        (1.0 if (series_state[(s, r)]["period"] <= 1 or (t % series_state[(s, r)]["period"]) == series_state[(s, r)]["phase"]) else 0.0)
        if (s, r) in series_state else 1.0
        for s, r, t in zip(ml_hist["SKU_ID"], ml_hist["Region"], ml_hist["t_int"])
    ]
    # Two-Stage Hurdle Global ML:
    # Train Stage-2 size model strictly on positive demand months (y > 0) using positive-month baseline
    # to eliminate zero-inflation Jensen bias, plus Duan smearing correction!
    ml_hist["pos_y"] = np.where(ml_hist["y"] > 0, ml_hist["y"], np.nan)
    ml_hist["feat_base_level"] = grp_ml["pos_y"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=2).mean()
    )
    ml_train = ml_hist[(ml_hist["y"] > 0) & ml_hist["feat_base_level"].notna() & (ml_hist["feat_base_level"] > 0)].copy()
    weights = np.where(ml_train["Is_One_Off_Spike"], 0.20, 1.0)
    X_tr = make_ml_features(ml_train, is_train=True)
    y_tr = np.log1p(ml_train["y"].values)

    reg_lambda = 40.0
    XtW = X_tr.T * weights
    A = XtW @ X_tr + reg_lambda * np.eye(X_tr.shape[1])
    A[0, 0] -= reg_lambda
    b = XtW @ y_tr
    w_ml = np.linalg.solve(A, b)
    # Duan smearing calibration factor on training positives
    tr_pred_raw = np.expm1(np.clip(X_tr @ w_ml, 0.0, 10.0))
    smear_factor = float(np.clip(np.sum(ml_train["y"].values * weights) / (np.sum(tr_pred_raw * weights) + 1e-6), 0.92, 1.12))

    # 4. Predict on future_keys
    fut = future_keys.copy().reset_index(drop=True)
    fut["t_int"] = fut["Date"].map(month_to_int)
    fut["Horizon"] = fut["t_int"] - origin_t
    fut["Month"] = fut["Date"].str[5:7].astype(int)
    fut["Year"] = fut["Date"].str[:4].astype(int)
    fut = fut.merge(sm, on="SKU_ID", how="left")

    # Populate series-level features and base model predictions for each future row
    n_fut = len(fut)
    pred_snaive = np.zeros(n_fut)
    pred_ma6 = np.zeros(n_fut)
    pred_tsb_cyc = np.zeros(n_fut)
    feat_base_level = np.zeros(n_fut)
    feat_seas_idx = np.ones(n_fut)
    feat_on_cycle = np.ones(n_fut)
    feat_nz_rate = np.ones(n_fut)
    sbc_list = []
    model_basis = []
    lifecycle_note = []
    unit_price_proj = np.zeros(n_fut)

    # Helper to project unit price from 2026 baseline using EV-059 (+4.5% 2027) and EV-065 (+4.0% 2028)
    def project_price(base_p: float, yr: int) -> float:
        p = base_p
        if yr >= 2027:
            p *= 1.045
        if yr >= 2028:
            p *= 1.040
        return round(p, 2)

    for i, r in enumerate(fut.itertuples(index=False)):
        key = (r.SKU_ID, r.Region)
        mm = int(r.Month)
        t_i = int(r.t_int)
        h_i = int(r.Horizon)
        yr = int(r.Year)

        # Case A: Planned Launch with NO history (Replaces_SKU!)
        if key not in series_state and pd.notna(r.Replaces_SKU):
            old_key = (r.Replaces_SKU, r.Region)
            st = series_state.get(old_key)
            if st is None:
                # Fallback to average across predecessor's regions
                old_states = [v for (s0, _), v in series_state.items() if s0 == r.Replaces_SKU]
                st = old_states[0] if old_states else None

            launch_t = month_to_int(r.Launch_Date) if pd.notna(r.Launch_Date) else t_i
            age = t_i - launch_t
            if st is None or age < 0:
                sbc_list.append("Planned Launch")
                model_basis.append("Pre-Launch Zero")
                lifecycle_note.append(f"Pre-launch (Launch={r.Launch_Date})")
                unit_price_proj[i] = project_price(st["last_price"] if st else 500.0, yr)
                continue

            ramp = ramp_up_curve.get(age, 1.0) if age < 6 else 1.0
            seas = st["own_seas"].get(mm, 1.0)
            base_val = st["ma12"] * seas * ramp
            pred_snaive[i] = st["snaive_map"].get(mm, st["ma12"]) * ramp
            pred_ma6[i] = st["ma6"] * ramp
            pred_tsb_cyc[i] = st["rate_tsb"] * seas * ramp
            feat_base_level[i] = st["ma12"] * ramp
            feat_seas_idx[i] = seas
            feat_on_cycle[i] = 1.0
            feat_nz_rate[i] = st["nz12"]
            sbc_list.append("Planned Launch")
            model_basis.append(f"Replaces_SKU({r.Replaces_SKU})*Ramp({ramp:.2f})")
            lifecycle_note.append(f"Replaces {r.Replaces_SKU}; month +{age} ramp={ramp:.2f}")
            unit_price_proj[i] = project_price(st["last_price"], yr)
            continue

        # Case B: Existing SKU-Region
        st = series_state.get(key)
        if st is None:
            sbc_list.append("Inactive/Zero")
            model_basis.append("No History")
            lifecycle_note.append("No history")
            unit_price_proj[i] = 500.0
            continue

        unit_price_proj[i] = project_price(st["last_price"], yr)
        sbc_list.append(st["sbc"])

        # Hard EOL check: no sales after EOL_Date!
        eol_mult = 1.0
        note = "Active"
        if pd.notna(r.EOL_Date):
            eol_t = month_to_int(r.EOL_Date)
            rem = eol_t - t_i
            if rem < 0:
                eol_mult = 0.0
                note = f"Post-EOL (EOL={r.EOL_Date})"
            elif rem < 3:
                eol_mult = min(1.0, ramp_down_curve.get(rem, 0.4 + 0.25 * rem))
                note = f"EOL Ramp-Down ({rem}m to {r.EOL_Date})"
            else:
                note = f"EOL Announced ({r.EOL_Date})"

        # Lost account suppression
        if st["is_lost"]:
            eol_mult = 0.0
            note = "Lost Account (6m+ zero)"

        seas = st["own_seas"].get(mm, 1.0)
        # Seasonal Naive
        pred_snaive[i] = st["snaive_map"].get(mm, st["ma12"]) * eol_mult
        # MA-6
        pred_ma6[i] = st["ma6"] * eol_mult

        # Cycle-aware TSB
        p_cyc = st["period"]
        if p_cyc >= 2 and st["cyc_str"] >= 0.55:
            is_on = 1.0 if (t_i % p_cyc) == st["phase"] else 0.0
            # On-cycle month gets z_tsb * seas, off-cycle gets tiny residual probability
            p_eff = min(0.95, st["p_tsb"] * p_cyc) if is_on > 0.5 else max(0.02, st["p_tsb"] * 0.08)
            tsb_val = p_eff * st["z_tsb"] * seas
            feat_on_cycle[i] = is_on
            basis = f"TSB+Cycle(p={p_cyc})+ML"
        else:
            tsb_val = st["rate_tsb"] * seas
            feat_on_cycle[i] = 1.0
            basis = f"Ensemble({st['sbc']})"

        # Subtle damped structural powertrain drift (EV growing, ICE declining modestly)
        drift = (1.0 + 0.45 * st["pt_growth"]) ** (min(h_i, 12) / 12.0)
        pred_tsb_cyc[i] = tsb_val * drift * eol_mult

        feat_base_level[i] = st["ma12"] * drift * eol_mult
        feat_seas_idx[i] = seas
        feat_nz_rate[i] = st["nz12"]
        model_basis.append(basis)
        lifecycle_note.append(note)

    fut["feat_base_level"] = feat_base_level
    fut["feat_seas_idx"] = feat_seas_idx
    fut["feat_on_cycle"] = feat_on_cycle
    fut["feat_nz_rate"] = feat_nz_rate
    fut["SBC_Class"] = sbc_list
    fut["Model_Basis"] = model_basis
    fut["Lifecycle_Note"] = lifecycle_note
    fut["Unit_Price_Projected"] = unit_price_proj

    # Global ML prediction (Stage 2 size * Duan smearing * Stage 1 cycle-aware order probability)
    # Convert feat_base_level (unconditional) to conditional positive size for Stage-2 input
    p_order_stage1 = np.where(
        feat_on_cycle < 0.5,
        np.clip(feat_nz_rate * 0.08, 0.0, 0.15),
        np.clip(feat_nz_rate / np.where(feat_on_cycle > 0.5, np.maximum(feat_nz_rate, 0.35), 1.0), 0.0, 1.0),
    )
    # Use actual unconditional probability for smooth/monthly, cycle-boosted for periodic
    p_order_eff = np.where(
        fut["Model_Basis"].str.startswith("TSB+Cycle").values,
        np.where(feat_on_cycle > 0.5, np.minimum(0.95, feat_nz_rate * 2.2), 0.03),
        feat_nz_rate,
    )
    cond_base = np.where(feat_nz_rate > 0.05, feat_base_level / np.maximum(feat_nz_rate, 0.15), feat_base_level)
    fut["feat_base_level"] = cond_base
    X_fut = make_ml_features(fut, is_train=False)
    pred_ml_size = np.expm1(np.clip(X_fut @ w_ml, 0.0, 10.0)) * smear_factor
    pred_ml = pred_ml_size * p_order_eff

    zero_force = (feat_base_level <= 0.0)
    pred_ml = np.where(zero_force, 0.0, pred_ml)

    # Apply EV-062 (Orion Driveline | North America planned retooling shutdown 2027-07..2027-08)
    orion_shutdown = (
        (fut["Customer_Name"] == "Orion Driveline")
        & (fut["Region"] == "North America")
        & (fut["Date"].isin(["2027-07", "2027-08"]))
    ).values
    pred_ml = np.where(orion_shutdown, pred_ml * 0.45, pred_ml)
    pred_tsb_cyc = np.where(orion_shutdown, pred_tsb_cyc * 0.45, pred_tsb_cyc)
    fut.loc[orion_shutdown, "Lifecycle_Note"] = "EV-062: Orion NA Retooling Shutdown (-55%)"

    # SBC-aware Ensemble Blend
    pred_ens = np.zeros(n_fut)
    for i in range(n_fut):
        sbc = sbc_list[i]
        if sbc == "Smooth":
            pred_ens[i] = 0.55 * pred_ml[i] + 0.35 * pred_tsb_cyc[i] + 0.10 * pred_snaive[i]
        elif sbc == "Erratic":
            pred_ens[i] = 0.50 * pred_ml[i] + 0.40 * pred_tsb_cyc[i] + 0.10 * pred_snaive[i]
        elif sbc in ("Intermittent", "Lumpy"):
            pred_ens[i] = 0.30 * pred_ml[i] + 0.70 * pred_tsb_cyc[i]
        elif sbc == "Planned Launch":
            pred_ens[i] = 0.45 * pred_ml[i] + 0.55 * pred_tsb_cyc[i]
        else:
            pred_ens[i] = 0.0

    fut["pred_snaive"] = np.clip(pred_snaive, 0.0, None)
    fut["pred_ma6"] = np.clip(pred_ma6, 0.0, None)
    fut["pred_tsb_cycle"] = np.clip(pred_tsb_cyc, 0.0, None)
    fut["pred_global_ml"] = np.clip(pred_ml, 0.0, None)
    fut["pred_ensemble_raw"] = np.clip(pred_ens, 0.0, None)

    return fut


def hierarchical_reconcile_integers(
    fut: pd.DataFrame, hist: pd.DataFrame, origin_date: str
) -> tuple[np.ndarray, dict]:
    """
    Middle-out / proportional hierarchical reconciliation at (Date, Product_Family, Region)
    so bottom-level integer forecasts add up EXACTLY to Family-Region, Family, Region, BU, and Global totals.
    """
    # Compute direct (Product_Family, Region) seasonal benchmark from last 12m to guard against bottom-up drift
    origin_t = month_to_int(origin_date)
    h12 = hist[hist["Date"].map(month_to_int) > origin_t - 12]
    fr_m_hist = h12.groupby(["Product_Family", "Region", "Month"])["Unconstrained_Demand_For_Model"].sum().to_dict()

    raw = fut["pred_ensemble_raw"].values.copy()
    reconciled_float = raw.copy()

    # Lightly reconcile each (Date, Product_Family, Region) group (85% bottom-up, 15% direct family-region seasonal)
    # Only when no major EOL/Launch/Orion event distorts the group
    for (dt, fam, reg), idxs in fut.groupby(["Date", "Product_Family", "Region"]).groups.items():
        idx_arr = np.array(list(idxs))
        bu_sum = raw[idx_arr].sum()
        if bu_sum <= 0:
            continue
        mm = int(dt[5:7])
        direct_ref = fr_m_hist.get((fam, reg, mm), bu_sum)
        # Keep reconciliation conservative so SKU-level accuracy stays sharp
        target_sum = 0.90 * bu_sum + 0.10 * direct_ref
        scale = np.clip(target_sum / bu_sum, 0.92, 1.08)
        reconciled_float[idx_arr] = raw[idx_arr] * scale

    # Largest-remainder integer rounding per (Date, Product_Family, Region) so integer sums match rounded group target
    reconciled_int = np.zeros(len(fut), dtype=int)
    for (dt, fam, reg), idxs in fut.groupby(["Date", "Product_Family", "Region"]).groups.items():
        idx_arr = np.array(list(idxs))
        vals = reconciled_float[idx_arr]
        target_int = int(round(vals.sum()))
        floors = np.floor(vals).astype(int)
        rem = target_int - int(floors.sum())
        if rem > 0:
            frac = vals - floors
            order = np.argsort(-frac)
            floors[order[:rem]] += 1
        elif rem < 0:
            frac = vals - floors
            pos_mask = floors > 0
            order = np.argsort(np.where(pos_mask, frac, 2.0))
            floors[order[: (-rem)]] -= 1
        reconciled_int[idx_arr] = np.clip(floors, 0, None)

    # Verify exact hierarchical additivity
    chk = fut[["Date", "Product_Family", "Business_Unit", "Region"]].copy()
    chk["y"] = reconciled_int
    tot_bottom = int(chk["y"].sum())
    tot_fam = int(chk.groupby("Product_Family")["y"].sum().sum())
    tot_reg = int(chk.groupby("Region")["y"].sum().sum())
    tot_bu = int(chk.groupby("Business_Unit")["y"].sum().sum())
    tot_dt = int(chk.groupby("Date")["y"].sum().sum())

    recon_report = {
        "method": "Proportional Middle-Out (Family-Region) + Largest-Remainder Coherent Integer Rounding",
        "total_bottom_up_units": tot_bottom,
        "total_by_family_sum": tot_fam,
        "total_by_region_sum": tot_reg,
        "total_by_bu_sum": tot_bu,
        "total_by_date_sum": tot_dt,
        "max_hierarchy_discrepancy": max(
            abs(tot_bottom - tot_fam),
            abs(tot_bottom - tot_reg),
            abs(tot_bottom - tot_bu),
            abs(tot_bottom - tot_dt),
        ),
    }
    return reconciled_int, recon_report


def calc_wape_bias(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    denom = float(np.sum(np.abs(y_true)))
    if denom <= 0:
        return 0.0, 0.0
    wape = float(np.sum(np.abs(y_true - y_pred)) / denom) * 100.0
    bias = float(np.sum(y_pred - y_true) / denom) * 100.0
    return round(wape, 2), round(bias, 2)


def run_forecasting(raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> dict:
    train_df = pd.read_csv(out_dir / "train_enriched.csv.gz", low_memory=False)
    test_df = pd.read_csv(raw_dir / "test.csv")
    macro = pd.read_csv(out_dir / "macro_clean.csv")
    sku_master = pd.read_csv(raw_dir / "sku_master.csv")
    events = pd.read_csv(raw_dir / "events_calendar.csv")

    ramp_up, ramp_down = calibrate_replacement_curves(train_df, sku_master)

    # --- 1. Rolling-Origin 19-Month Backtest (2 origins) ---
    origins = [
        ("2024-09", "2024-10", "2026-04"),
        ("2025-03", "2025-04", "2026-10"),
    ]
    bt_frames = []
    for orig, start_dt, end_dt in origins:
        val_actual = train_df[
            (train_df["Date"] >= start_dt) & (train_df["Date"] <= end_dt)
        ].copy()
        val_keys = val_actual[
            ["Date", "Year", "Month", "Product_Family", "Business_Unit", "SKU_ID", "SKU_Description", "Customer_Name", "Sales_Channel", "Region"]
        ].copy()
        preds = fit_and_predict_origin(
            train_df, val_keys, macro, sku_master, events, orig, ramp_up, ramp_down
        )
        preds["y_true"] = val_actual["Unconstrained_Demand_For_Model"].values
        preds["Origin"] = orig
        bt_frames.append(preds)

    bt = pd.concat(bt_frames, ignore_index=True)

    # Overall model comparison
    models = {
        "Seasonal Naive (Baseline 1)": "pred_snaive",
        "Moving Average 6M (Baseline 2)": "pred_ma6",
        "TSB Cycle-Aware (Intermittent)": "pred_tsb_cycle",
        "Global ML (Ridge + Macro/Events)": "pred_global_ml",
        "Final Ensemble (SBC + Lifecycle)": "pred_ensemble_raw",
    }
    model_metrics = {}
    for m_name, col in models.items():
        w, b = calc_wape_bias(bt["y_true"].values, bt[col].values)
        model_metrics[m_name] = {"WAPE_Pct": w, "Bias_Pct": b}

    # Metrics by Family, Region, and Horizon (for Final Ensemble)
    by_family = {}
    for fam, g in bt.groupby("Product_Family"):
        w, b = calc_wape_bias(g["y_true"].values, g["pred_ensemble_raw"].values)
        w_sn, _ = calc_wape_bias(g["y_true"].values, g["pred_snaive"].values)
        by_family[fam] = {"WAPE_Pct": w, "Bias_Pct": b, "Baseline_SN_WAPE_Pct": w_sn}

    by_region = {}
    for reg, g in bt.groupby("Region"):
        w, b = calc_wape_bias(g["y_true"].values, g["pred_ensemble_raw"].values)
        w_sn, _ = calc_wape_bias(g["y_true"].values, g["pred_snaive"].values)
        by_region[reg] = {"WAPE_Pct": w, "Bias_Pct": b, "Baseline_SN_WAPE_Pct": w_sn}

    by_horizon = []
    for h, g in bt.groupby("Horizon"):
        w, b = calc_wape_bias(g["y_true"].values, g["pred_ensemble_raw"].values)
        w_sn, _ = calc_wape_bias(g["y_true"].values, g["pred_snaive"].values)
        by_horizon.append({"Horizon": int(h), "WAPE_Pct": w, "Bias_Pct": b, "Baseline_SN_WAPE_Pct": w_sn})

    # Calibrate empirical conformal P10 / P90 multipliers from backtest residuals by (SBC_Class, Horizon_Bucket)
    bt["H_Bucket"] = pd.cut(bt["Horizon"], bins=[0, 6, 12, 20], labels=["1-6", "7-12", "13-19"]).astype(str)
    bt["Rel_Err"] = (bt["y_true"] - bt["pred_ensemble_raw"]) / np.maximum(bt["pred_ensemble_raw"], 5.0)
    q_table = {}
    for (sbc, hb), g in bt.groupby(["SBC_Class", "H_Bucket"]):
        errs = g["Rel_Err"].values
        q10 = float(np.clip(np.quantile(errs, 0.10), -0.85, -0.05))
        q90 = float(np.clip(np.quantile(errs, 0.90), 0.10, 1.50))
        q_table[(sbc, hb)] = (q10, q90)

    # --- 2. Final 19-Month Forecast on test.csv (2026-11 .. 2028-05) ---
    test_preds = fit_and_predict_origin(
        train_df, test_df, macro, sku_master, events, "2026-10", ramp_up, ramp_down
    )

    # Hierarchical coherent integer reconciliation
    p50_int, recon_report = hierarchical_reconcile_integers(test_preds, train_df, "2026-10")
    test_preds["Units_Sold_Forecast"] = p50_int
    test_preds["P50"] = p50_int

    # Compute P10 and P90 using calibrated backtest quantiles
    h_buckets = pd.cut(test_preds["Horizon"], bins=[0, 6, 12, 20], labels=["1-6", "7-12", "13-19"]).astype(str)
    p10_arr = np.zeros(len(test_preds), dtype=int)
    p90_arr = np.zeros(len(test_preds), dtype=int)
    for i, (sbc, hb, p50) in enumerate(zip(test_preds["SBC_Class"], h_buckets, p50_int)):
        if p50 == 0:
            p10_arr[i] = 0
            p90_arr[i] = 0
            continue
        q10, q90 = q_table.get((sbc, hb), (-0.30, 0.35))
        p10_arr[i] = max(0, int(round(p50 * (1.0 + q10))))
        p90_arr[i] = max(p50, int(round(p50 * (1.0 + q90))))

    test_preds["P10"] = p10_arr
    test_preds["P90"] = p90_arr
    test_preds["Revenue_P50"] = np.round(test_preds["P50"] * test_preds["Unit_Price_Projected"], 2)
    test_preds["Revenue_P90"] = np.round(test_preds["P90"] * test_preds["Unit_Price_Projected"], 2)

    # Save submission.csv (exact test.csv key order, non-negative ints, no blanks)
    sub = test_preds[["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]].copy()
    sub["Units_Sold_Forecast"] = sub["Units_Sold_Forecast"].astype(int)
    assert len(sub) == len(test_df), f"Row count mismatch: {len(sub)} vs {len(test_df)}"
    assert sub.isna().sum().sum() == 0, "Submission has nulls!"
    assert (sub["Units_Sold_Forecast"] < 0).sum() == 0, "Submission has negative values!"
    sub.to_csv(out_dir / "submission.csv", index=False)

    # Save full forecast parquet for dashboard
    test_preds.to_csv(out_dir / "forecast_full.csv.gz", index=False)

    backtest_summary = {
        "origins": [
            {"origin": o, "start": s, "end": e, "horizon_months": 19}
            for o, s, e in origins
        ],
        "model_comparison": model_metrics,
        "by_family": by_family,
        "by_region": by_region,
        "by_horizon": by_horizon,
        "calibrated_ramp_up_curve": ramp_up,
        "calibrated_ramp_down_curve": ramp_down,
        "reconciliation": recon_report,
        "forecast_headline": {
            "total_rows": len(sub),
            "total_p10_units": int(test_preds["P10"].sum()),
            "total_p50_units": int(test_preds["P50"].sum()),
            "total_p90_units": int(test_preds["P90"].sum()),
            "total_p50_revenue_usd": round(float(test_preds["Revenue_P50"].sum()), 2),
            "total_p90_revenue_usd": round(float(test_preds["Revenue_P90"].sum()), 2),
        },
    }
    with open(out_dir / "backtest_metrics.json", "w") as f:
        json.dump(backtest_summary, f, indent=2)
    with open(out_dir / "reconciliation_check.json", "w") as f:
        json.dump(recon_report, f, indent=2)

    return backtest_summary


if __name__ == "__main__":
    summary = run_forecasting()
    print(json.dumps(summary["model_comparison"], indent=2))
    print("\nForecast Headline:")
    print(json.dumps(summary["forecast_headline"], indent=2))
    print("\nReconciliation:")
    print(json.dumps(summary["reconciliation"], indent=2))
