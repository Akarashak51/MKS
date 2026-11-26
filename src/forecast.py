import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUT_DIR, RAW_DIR
from features_drivers import detect_cycle_and_phase


def month_to_int(ym: str) -> int:
    y, m = ym.split("-")
    return int(y) * 12 + int(m)


def build_holiday_flags(dates: pd.Series, regions: pd.Series, events: pd.DataFrame) -> pd.DataFrame:
    lny_months, ramadan_months = set(), set()
    for r in events.itertuples(index=False):
        desc = str(r.Description).lower()
        sd = str(r.Start_Date)[:7]
        ed = str(r.End_Date)[:7] if pd.notna(r.End_Date) else sd
        if "lunar new year" in desc:
            lny_months.update([sd, ed])
        elif "ramadan" in desc:
            ramadan_months.update([sd, ed])

    m_num = dates.str[5:7].astype(int)
    out = pd.DataFrame(index=dates.index)
    out["is_NA_July"] = ((regions == "North America") & (m_num == 7)).astype(float)
    out["is_EU_Aug"] = ((regions == "Europe") & (m_num == 8)).astype(float)
    out["is_APAC_LNY"] = ((regions == "Asia Pacific") & (dates.isin(lny_months))).astype(float)
    out["is_MEA_Ramadan"] = ((regions == "Middle East & Africa") & (dates.isin(ramadan_months))).astype(float)
    return out


def build_price_hike_map(events: pd.DataFrame) -> dict[str, float]:
    hike_map = {}
    for r in events.itertuples(index=False):
        if r.Event_Type == "Price Change" and "1 January" in str(r.Description):
            m = re.search(r"(\d+\.?\d*)%", str(r.Description))
            if m:
                hike_map[str(r.Start_Date)[:7]] = float(m.group(1))
    return hike_map


def calibrate_replacement_curves(df: pd.DataFrame, sku_master: pd.DataFrame) -> tuple[dict, dict]:
    rep_pairs = sku_master[sku_master["Replaces_SKU"].notna() & (sku_master["Lifecycle_Status"] == "Active")]
    ramp_up_samples = {m: [] for m in range(6)}
    ramp_down_samples = {m: [] for m in range(4)}

    for r in rep_pairs.itertuples(index=False):
        new_sku, old_sku = r.SKU_ID, r.Replaces_SKU
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

    ramp_up = {m: round(float(np.median(v)) if v else min(1.0, 0.35 + 0.2 * m), 3) for m, v in ramp_up_samples.items()}
    ramp_down = {m: round(float(np.median(v)) if v else (0.35 + 0.2 * m), 3) for m, v in ramp_down_samples.items()}
    return ramp_up, ramp_down


def tsb_forecast_series(y: np.ndarray, alpha_d: float = 0.18, alpha_p: float = 0.18) -> tuple[float, float, float]:
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


def apply_reconciliation_and_rounding(
    fut: pd.DataFrame, hist: pd.DataFrame, origin_date: str, pred_col: str = "pred_ensemble_raw"
) -> tuple[np.ndarray, dict]:
    """
    90/10 proportional pull toward prior-year same-month Family-Region volume (clipped to +/-8%)
    followed by largest-remainder integer rounding.
    """
    origin_t = month_to_int(origin_date)
    h12 = hist[hist["Date"].map(month_to_int) > origin_t - 12]
    fr_m_hist = h12.groupby(["Product_Family", "Region", "Month"])["Unconstrained_Demand_For_Model"].sum().to_dict()

    raw = fut[pred_col].values.copy()
    reconciled_float = raw.copy()

    for (dt, fam, reg), idxs in fut.groupby(["Date", "Product_Family", "Region"]).groups.items():
        idx_arr = np.array(list(idxs))
        bu_sum = raw[idx_arr].sum()
        if bu_sum <= 0:
            continue
        mm = int(dt[5:7])
        direct_ref = fr_m_hist.get((fam, reg, mm), bu_sum)
        target_sum = 0.90 * bu_sum + 0.10 * direct_ref
        scale = np.clip(target_sum / bu_sum, 0.92, 1.08)
        reconciled_float[idx_arr] = raw[idx_arr] * scale

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
            order = np.argsort(np.where(floors > 0, frac, 2.0))
            floors[order[: (-rem)]] -= 1
        reconciled_int[idx_arr] = np.clip(floors, 0, None)

    recon_report = {
        "method": "90/10 proportional pull toward prior-year same-month Family-Region volume (clipped +/-8%) + largest-remainder integer rounding",
        "total_bottom_up_units": int(reconciled_int.sum()),
        "raw_ensemble_sum_units": round(float(raw.sum()), 1),
        "net_reconciliation_adjustment_pct": round(100.0 * (float(reconciled_int.sum()) / max(1.0, float(raw.sum())) - 1.0), 3),
        "max_hierarchy_discrepancy": 0,
        "additivity_note": "Bottom-up integer rows sum identically across SKU -> Family -> BU -> Region -> Global groupings by construction.",
    }
    return reconciled_int, recon_report


def fit_and_predict_origin(
    train_df: pd.DataFrame,
    future_keys: pd.DataFrame,
    macro: pd.DataFrame,
    sku_master: pd.DataFrame,
    events: pd.DataFrame,
    origin_date: str,
    ramp_up_curve: dict,
    ramp_down_curve: dict,
    family_best_lags: dict,
) -> tuple[pd.DataFrame, dict]:
    hist = train_df[train_df["Date"] <= origin_date].copy()
    origin_t = month_to_int(origin_date)
    hist["t_int"] = hist["Date"].map(month_to_int)
    hist["y"] = hist["Unconstrained_Demand_For_Model"].fillna(0.0).clip(lower=0.0)

    sm = sku_master[["SKU_ID", "Powertrain_Application", "Launch_Date", "EOL_Date", "Replaces_SKU", "Lifecycle_Status"]].copy()
    hist = hist.merge(sm, on="SKU_ID", how="left")

    # Regional-family seasonal indices (non-COVID)
    clean_seas = hist[~hist["Is_Covid_Disruption"] & (hist["y"] > 0)]
    rf_month_mean = clean_seas.groupby(["Product_Family", "Region", "Month"])["y"].mean()
    rf_overall_mean = clean_seas.groupby(["Product_Family", "Region"])["y"].mean()
    rf_seas_idx = (rf_month_mean / (rf_overall_mean + 1e-6)).clip(0.55, 1.55).to_dict()

    last_12_start = origin_t - 11
    prev_12_start = origin_t - 23
    pt_last12 = hist[hist["t_int"].between(last_12_start, origin_t)].groupby(["Product_Family", "Powertrain_Application"])["y"].mean()
    pt_prev12 = hist[hist["t_int"].between(prev_12_start, last_12_start - 1)].groupby(["Product_Family", "Powertrain_Application"])["y"].mean()
    pt_growth_annual = ((pt_last12 / (pt_prev12 + 1e-6)) - 1.0).clip(-0.18, 0.18).to_dict()

    # Series state strictly <= origin_date
    series_state = {}
    for (sid, reg), g in hist.groupby(["SKU_ID", "Region"], sort=False):
        g = g.sort_values("t_int")
        y_vals = g["y"].values

        last24 = g[g["t_int"] > origin_t - 24]
        last12 = g[g["t_int"] > origin_t - 12]
        last6 = g[g["t_int"] > origin_t - 6]

        y24, t24 = last24["y"].values, last24["t_int"].values
        y12, y6 = last12["y"].values, last6["y"].values

        pos24 = y24[y24 > 0]
        med_pos = float(np.median(pos24)) if len(pos24) > 0 else 0.0
        y12_wins = np.minimum(y12, 2.5 * med_pos) if med_pos > 0 else y12

        ma6 = float(y6.mean()) if len(y6) > 0 else 0.0
        ma12 = float(y12_wins.mean()) if len(y12_wins) > 0 else ma6
        nz12 = float((y12 > 0).mean()) if len(y12) > 0 else 0.0

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

        is_lost = bool((y_vals > 0).sum() >= 6 and len(y_vals) >= 12 and np.all(y_vals[-6:] == 0))
        period, phase, cyc_str = detect_cycle_and_phase(y24, t24)
        p_tsb, z_tsb, rate_tsb = tsb_forecast_series(y_vals[-36:])

        snaive_map = {int(mm): float(yy) for mm, yy in zip(last12["Month"].values, last12["y"].values)}
        fam, pt = g["Product_Family"].iloc[0], g["Powertrain_Application"].iloc[0]

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

        cust_latest = g["Customer_Name"].iloc[-1]
        if cust_latest == "Helios Auto Parts" and reg == "India" and origin_date >= "2025-11":
            post_h = g[g["Date"] >= "2025-09"]["y"].values
            if len(post_h) >= 3:
                ma12 = float(post_h.mean())
                z_tsb = float(post_h[post_h > 0].mean()) if (post_h > 0).any() else z_tsb

        c_arr = g["Customer_Name"].values
        sw = np.where(c_arr[1:] != c_arr[:-1])[0]
        if len(sw) > 0:
            last_sw_idx = sw[-1] + 1
            if 3 <= (len(g) - last_sw_idx) <= 18:
                post_sw = y_vals[last_sw_idx:]
                ma12 = float(post_sw.mean())
                p_tsb, z_tsb, rate_tsb = tsb_forecast_series(post_sw)

        series_state[(sid, reg)] = {
            "ma6": ma6, "ma12": ma12, "p_tsb": p_tsb, "z_tsb": z_tsb, "rate_tsb": rate_tsb,
            "nz12": nz12, "sbc": sbc, "is_lost": is_lost, "period": period, "phase": phase,
            "cyc_str": cyc_str, "snaive_map": snaive_map, "own_seas": own_seas, "fam": fam,
            "pt": pt, "pt_growth": float(pt_growth_annual.get((fam, pt), 0.0)),
            "last_price": float(g["Unit_Price_USD"].dropna().iloc[-1]) if g["Unit_Price_USD"].notna().any() else 500.0,
        }

    # Prepare macro indicators with lags 0..6 standardized strictly on history <= origin_date
    m_df = macro.sort_values("Date").copy()
    macro_cols = [
        "ISM_Manufacturing_PMI",
        "Auto_Production_YoY_Pct",
        "Vehicle_Inventory_Days",
        "New_Vehicle_Orders_Index",
        "Industrial_Util_Rate_Pct",
        "EV_Adoption_Index",
    ]
    m_hist = m_df[m_df["Date"] <= origin_date]
    m_mean = {c: float(m_hist[c].mean()) for c in macro_cols}
    m_std = {c: float(m_hist[c].std()) or 1.0 for c in macro_cols}

    # Build lookup: (Date, Indicator, Lag) -> z-score
    macro_z_lookup = {}
    for c in macro_cols:
        z_series = (m_df[c] - m_mean[c]) / m_std[c]
        for lag in range(7):
            shifted = z_series.shift(lag).bfill()
            macro_z_lookup[(c, lag)] = dict(zip(m_df["Date"].values, shifted.values))

    hike_map = build_price_hike_map(events)

    def get_fam_lagged_macro(dates_arr, fams_arr, ind: str) -> np.ndarray:
        out = np.zeros(len(dates_arr))
        for i, (dt, fam) in enumerate(zip(dates_arr, fams_arr)):
            lag = int(family_best_lags.get(fam, {}).get(ind, 2))
            lag = max(0, min(6, lag))
            out[i] = macro_z_lookup[(ind, lag)].get(dt, 0.0)
        return out

    def make_ml_features(frame: pd.DataFrame, is_train: bool) -> np.ndarray:
        hol = build_holiday_flags(frame["Date"], frame["Region"], events)
        mm = frame["Month"].astype(float).values
        sin_m, cos_m = np.sin(2.0 * np.pi * mm / 12.0), np.cos(2.0 * np.pi * mm / 12.0)

        dts = frame["Date"].values
        fams = frame["Product_Family"].values
        next_ym = [
            f"{int(d[:4]) + (1 if int(d[5:7]) == 12 else 0):04d}-{(int(d[5:7]) % 12) + 1:02d}"
            for d in dts
        ]
        pre_hike = np.array([hike_map.get(ny, 0.0) / 10.0 for ny in next_ym])
        post_hike = np.array([hike_map.get(d, 0.0) / 10.0 for d in dts])

        base_level = np.log1p(frame["feat_base_level"].fillna(0.0).clip(lower=0.0).values)
        seas_log = np.log(frame["feat_seas_idx"].fillna(1.0).clip(0.4, 2.0).values)
        on_cyc = frame["feat_on_cycle"].fillna(1.0).values
        nz_rate = frame["feat_nz_rate"].fillna(1.0).values

        # Family-specific optimal lags from Workstream 3!
        pmi_z = get_fam_lagged_macro(dts, fams, "ISM_Manufacturing_PMI")
        auto_z = get_fam_lagged_macro(dts, fams, "Auto_Production_YoY_Pct")
        ord_z = get_fam_lagged_macro(dts, fams, "New_Vehicle_Orders_Index")
        inv_z = get_fam_lagged_macro(dts, fams, "Vehicle_Inventory_Days")
        util_z = get_fam_lagged_macro(dts, fams, "Industrial_Util_Rate_Pct")
        ev_z = get_fam_lagged_macro(dts, fams, "EV_Adoption_Index")

        is_ev = (frame["Powertrain_Application"] == "EV").astype(float).values
        is_ice = (frame["Powertrain_Application"] == "ICE").astype(float).values

        if is_train:
            is_cov = frame["Is_Covid_Disruption"].astype(float).values
            is_nex = frame["Is_Nexon_Strike"].astype(float).values
            is_red = frame["Is_Red_Sea"].astype(float).values
        else:
            is_cov = np.zeros(len(frame))
            is_nex = np.zeros(len(frame))
            is_red = np.zeros(len(frame))

        fam_pmi = []
        for fam_name in [
            "Brake & Safety Systems", "Chassis & Structural", "Electrical Assemblies",
            "Hydraulic Systems", "Powertrain Components", "Precision Machined Parts",
        ]:
            fm = (fams == fam_name).astype(float)
            fam_pmi.append(fm)
            fam_pmi.append(fm * pmi_z)

        cols = [
            np.ones(len(frame)),
            base_level, seas_log, on_cyc, nz_rate, base_level * on_cyc,
            sin_m, cos_m,
            hol["is_NA_July"].values, hol["is_EU_Aug"].values, hol["is_APAC_LNY"].values, hol["is_MEA_Ramadan"].values,
            pre_hike, post_hike,
            pmi_z, auto_z, ord_z, inv_z, util_z, ev_z,
            is_ev, is_ice, is_ev * ev_z, is_ice * ev_z,
            is_cov, is_nex, is_red,
        ] + fam_pmi
        return np.column_stack(cols)

    # Fit Ridge regression on trailing 42 months of positive demand + Duan smearing
    ml_hist = hist[hist["t_int"] > origin_t - 42].copy().sort_values(["SKU_ID", "Region", "t_int"])
    grp_ml = ml_hist.groupby(["SKU_ID", "Region"], sort=False)
    ml_hist["pos_y"] = np.where(ml_hist["y"] > 0, ml_hist["y"], np.nan)
    ml_hist["feat_base_level"] = grp_ml["pos_y"].transform(lambda s: s.shift(1).rolling(12, min_periods=2).mean())
    ml_hist["feat_nz_rate"] = grp_ml["y"].transform(lambda s: (s.shift(1) > 0).astype(float).rolling(12, min_periods=3).mean())
    ml_hist["feat_seas_idx"] = [
        rf_seas_idx.get((fam, reg, int(m)), 1.0)
        for fam, reg, m in zip(ml_hist["Product_Family"], ml_hist["Region"], ml_hist["Month"])
    ]
    ml_hist["feat_on_cycle"] = [
        (1.0 if (series_state[(s, r)]["period"] <= 1 or (t % series_state[(s, r)]["period"]) == series_state[(s, r)]["phase"]) else 0.0)
        if (s, r) in series_state else 1.0
        for s, r, t in zip(ml_hist["SKU_ID"], ml_hist["Region"], ml_hist["t_int"])
    ]
    ml_train = ml_hist[(ml_hist["y"] > 0) & ml_hist["feat_base_level"].notna() & (ml_hist["feat_base_level"] > 0)].copy()
    weights = np.where(ml_train["Is_One_Off_Spike"], 0.20, 1.0)
    X_tr = make_ml_features(ml_train, is_train=True)
    y_tr = np.log1p(ml_train["y"].values)

    reg_lambda = 40.0
    XtW = X_tr.T * weights
    A = XtW @ X_tr + reg_lambda * np.eye(X_tr.shape[1])
    A[0, 0] -= reg_lambda
    w_ml = np.linalg.solve(A, XtW @ y_tr)
    tr_pred_raw = np.expm1(np.clip(X_tr @ w_ml, 0.0, 10.0))
    smear_factor = float(np.clip(np.sum(ml_train["y"].values * weights) / (np.sum(tr_pred_raw * weights) + 1e-6), 0.92, 1.12))

    # Extract model-derived What-If elasticities directly from Ridge weights w_ml!
    # Indices in cols: 14=pmi_z, 15=auto_z, 19=ev_z, 22=is_ev*ev_z, 23=is_ice*ev_z, 13=post_hike (per 10% price)
    elasticities = {
        "pmi_pct_per_pt": round(float(w_ml[14] / m_std["ISM_Manufacturing_PMI"]), 5),
        "auto_yoy_pct_per_pct": round(float(w_ml[15] / m_std["Auto_Production_YoY_Pct"]), 5),
        "ev_universal_pct_per_pt": round(float(w_ml[19] / m_std["EV_Adoption_Index"]), 5),
        "ev_sku_pct_per_pt": round(float((w_ml[19] + w_ml[22]) / m_std["EV_Adoption_Index"]), 5),
        "ice_sku_pct_per_pt": round(float((w_ml[19] + w_ml[23]) / m_std["EV_Adoption_Index"]), 5),
        "price_pct_per_1pct": round(float(w_ml[13] / 10.0), 5),
        "source": "Derived directly from fitted Ridge regression weights w_ml divided by macro training std dev",
    }

    # Predict on future_keys
    fut = future_keys.copy().reset_index(drop=True)
    fut["t_int"] = fut["Date"].map(month_to_int)
    fut["Horizon"] = fut["t_int"] - origin_t
    fut["Month"] = fut["Date"].str[5:7].astype(int)
    fut["Year"] = fut["Date"].str[:4].astype(int)
    fut = fut.merge(sm, on="SKU_ID", how="left")

    n_fut = len(fut)
    pred_snaive = np.zeros(n_fut)
    pred_ma6 = np.zeros(n_fut)
    pred_tsb_cyc = np.zeros(n_fut)
    feat_base_level = np.zeros(n_fut)
    feat_seas_idx = np.ones(n_fut)
    feat_on_cycle = np.ones(n_fut)
    feat_nz_rate = np.ones(n_fut)
    sbc_list, model_basis, lifecycle_note = [], [], []
    unit_price_proj = np.zeros(n_fut)

    def project_price(base_p: float, yr: int) -> float:
        p = base_p
        if yr >= 2027: p *= 1.045
        if yr >= 2028: p *= 1.040
        return round(p, 2)

    for i, r in enumerate(fut.itertuples(index=False)):
        key = (r.SKU_ID, r.Region)
        mm, t_i, h_i, yr = int(r.Month), int(r.t_int), int(r.Horizon), int(r.Year)

        if key not in series_state and pd.notna(r.Replaces_SKU):
            st = series_state.get((r.Replaces_SKU, r.Region))
            if st is None:
                olds = [v for (s0, _), v in series_state.items() if s0 == r.Replaces_SKU]
                st = olds[0] if olds else None
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

        st = series_state.get(key)
        if st is None:
            sbc_list.append("Inactive/Zero")
            model_basis.append("No History")
            lifecycle_note.append("No history")
            unit_price_proj[i] = 500.0
            continue

        unit_price_proj[i] = project_price(st["last_price"], yr)
        sbc_list.append(st["sbc"])

        eol_mult, note = 1.0, "Active"
        if pd.notna(r.EOL_Date):
            rem = month_to_int(r.EOL_Date) - t_i
            if rem < 0:
                eol_mult, note = 0.0, f"Post-EOL (EOL={r.EOL_Date})"
            elif rem < 3:
                eol_mult = min(1.0, ramp_down_curve.get(rem, 0.4 + 0.25 * rem))
                note = f"EOL Ramp-Down ({rem}m to {r.EOL_Date})"
            else:
                note = f"EOL Announced ({r.EOL_Date})"

        if st["is_lost"]:
            eol_mult, note = 0.0, "Lost Account (6m+ zero)"

        seas = st["own_seas"].get(mm, 1.0)
        pred_snaive[i] = st["snaive_map"].get(mm, st["ma12"]) * eol_mult
        pred_ma6[i] = st["ma6"] * eol_mult

        p_cyc = st["period"]
        if p_cyc >= 2 and st["cyc_str"] >= 0.55:
            is_on = 1.0 if (t_i % p_cyc) == st["phase"] else 0.0
            p_eff = min(0.95, st["p_tsb"] * p_cyc) if is_on > 0.5 else max(0.02, st["p_tsb"] * 0.08)
            tsb_val = p_eff * st["z_tsb"] * seas
            feat_on_cycle[i] = is_on
            basis = f"TSB+Cycle(p={p_cyc})+Ridge"
        else:
            tsb_val = st["rate_tsb"] * seas
            feat_on_cycle[i] = 1.0
            basis = f"Ensemble({st['sbc']})"

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

    p_order_eff = np.where(
        fut["Model_Basis"].str.startswith("TSB+Cycle").values,
        np.where(feat_on_cycle > 0.5, np.minimum(0.95, feat_nz_rate * 2.2), 0.03),
        feat_nz_rate,
    )
    cond_base = np.where(feat_nz_rate > 0.05, feat_base_level / np.maximum(feat_nz_rate, 0.15), feat_base_level)
    fut["feat_base_level"] = cond_base
    X_fut = make_ml_features(fut, is_train=False)
    pred_ml = np.expm1(np.clip(X_fut @ w_ml, 0.0, 10.0)) * smear_factor * p_order_eff
    pred_ml = np.where(feat_base_level <= 0.0, 0.0, pred_ml)

    # Blend BEFORE applying forecast-window shutdown event so the full 55% cut applies cleanly
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

    # Apply FULL 55% cut for EV-062 (Orion Driveline | North America 2027-07..2027-08) across ALL predictions
    orion_shutdown = (
        (fut["Customer_Name"] == "Orion Driveline")
        & (fut["Region"] == "North America")
        & (fut["Date"].isin(["2027-07", "2027-08"]))
    ).values
    pred_snaive = np.where(orion_shutdown, pred_snaive * 0.45, pred_snaive)
    pred_tsb_cyc = np.where(orion_shutdown, pred_tsb_cyc * 0.45, pred_tsb_cyc)
    pred_ml = np.where(orion_shutdown, pred_ml * 0.45, pred_ml)
    pred_ens = np.where(orion_shutdown, pred_ens * 0.45, pred_ens)
    fut.loc[orion_shutdown, "Lifecycle_Note"] = "EV-062: Orion NA Retooling Shutdown (-55% applied post-blend)"

    fut["pred_snaive"] = np.clip(pred_snaive, 0.0, None)
    fut["pred_ma6"] = np.clip(pred_ma6, 0.0, None)
    fut["pred_tsb_cycle"] = np.clip(pred_tsb_cyc, 0.0, None)
    fut["pred_ridge_ml"] = np.clip(pred_ml, 0.0, None)
    fut["pred_ensemble_raw"] = np.clip(pred_ens, 0.0, None)

    return fut, elasticities


def calc_wape_bias(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    denom = float(np.sum(np.abs(y_true)))
    if denom <= 0:
        return 0.0, 0.0
    return round(float(np.sum(np.abs(y_true - y_pred)) / denom) * 100.0, 2), round(float(np.sum(y_pred - y_true) / denom) * 100.0, 2)


def calibrate_and_validate_intervals(bt: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    """
    Out-of-sample split-conformal calibration across ALL 5 hierarchy levels:
    - Calibrated strictly on Origins 1-3 (2023-09, 2024-03, 2024-09)
    - Validated strictly OUT-OF-SAMPLE on Origin 4 (2025-03 -> 2025-04..2026-10)
    Produces BOTH:
      1. P10-P90 Planning Band (centered split-conformal, target ~80% OOS at every level)
      2. P01-P99 Stress Envelope (tail-risk split-conformal, target ~99% OOS at every level)
    """
    bt = bt.copy()
    bt["H_Bucket"] = pd.cut(bt["Horizon"], bins=[0, 6, 12, 20], labels=["1-6", "7-12", "13-19"]).astype(str)
    cal = bt[bt["Origin"] != "2025-03"].copy()
    oos = bt[bt["Origin"] == "2025-03"].copy()

    # --- A. SKU-Region Level Calibration ---
    cal["Rel_Err"] = (cal["y_true"] - cal["pred_reconciled"]) / np.maximum(cal["pred_reconciled"], 2.0)
    raw_q80, raw_q99 = {}, {}
    for (sbc, hb), g in cal.groupby(["SBC_Class", "H_Bucket"]):
        errs = g["Rel_Err"].values
        raw_q80[(sbc, hb)] = (float(np.quantile(errs, 0.10)), float(np.quantile(errs, 0.90)))
        # Finite-sample 99% conformal quantiles
        n = len(errs)
        lo_p = max(0.0, 0.005 * (1.0 - 1.0 / max(1, n)))
        hi_p = min(1.0, 1.0 - lo_p)
        raw_q99[(sbc, hb)] = (float(np.quantile(errs, lo_p)), float(np.quantile(errs, hi_p)) * 1.15)

    # Tune conformal scale factor k on cal_va (Origin 3) so SKU P10-P90 targets 80% OOS
    cal_tr = cal[cal["Origin"] != "2024-09"]
    cal_va = cal[cal["Origin"] == "2024-09"]
    q_tr = {}
    for (sbc, hb), g in cal_tr.groupby(["SBC_Class", "H_Bucket"]):
        errs = g["Rel_Err"].values
        q_tr[(sbc, hb)] = (float(np.quantile(errs, 0.10)), float(np.quantile(errs, 0.90)))

    def eval_sku_band(frame, q_map, k_mult, is_99=False):
        p50 = frame["pred_reconciled"].values
        y = frame["y_true"].values
        p_lo, p_hi = np.zeros(len(frame)), np.zeros(len(frame))
        for i, (sbc, hb, v) in enumerate(zip(frame["SBC_Class"], frame["H_Bucket"], p50)):
            if v <= 0:
                p_lo[i], p_hi[i] = 0.0, (5.0 if is_99 else (1.0 if sbc in ("Intermittent", "Lumpy") else 0.0))
                continue
            ql, qh = q_map.get((sbc, hb), (-0.40, 0.45))
            p_lo[i] = max(0.0, np.floor(v * (1.0 + ql * k_mult)))
            p_hi[i] = max(v, np.ceil(v * (1.0 + qh * k_mult)))
        cov = float(((y >= p_lo) & (y <= p_hi)).mean()) * 100.0
        width = float(np.sum(p_hi - p_lo) / max(1.0, np.sum(y))) * 100.0
        return cov, width

    best_k, best_diff = 1.0, 999.0
    for k in np.linspace(0.95, 1.65, 29):
        cov_va, _ = eval_sku_band(cal_va, q_tr, k)
        if abs(cov_va - 80.0) < best_diff:
            best_diff, best_k = abs(cov_va - 80.0), float(k)

    sku_q80_final = {k: (round(v[0] * best_k, 4), round(v[1] * best_k, 4)) for k, v in raw_q80.items()}
    sku_q99_final = {k: (round(max(-1.0, v[0]), 4), round(v[1], 4)) for k, v in raw_q99.items()}
    sku_oos_cov80, sku_oos_w80 = eval_sku_band(oos, raw_q80, best_k, is_99=False)
    sku_oos_cov99, sku_oos_w99 = eval_sku_band(oos, raw_q99, 1.0, is_99=True)

    # --- B. Aggregate Level Calibration (Global, Family, Region, Family-Region) ---
    # Centered split-conformal residuals (removes growth-regime bias shift from Origins 1-2!)
    def fit_agg_level(group_cols: list[str], level_name: str) -> tuple[dict, dict]:
        cols_cal = ["Origin", "Date", "H_Bucket"] + group_cols
        g_cal = cal.groupby(cols_cal, as_index=False).agg(y=("y_true", "sum"), yhat=("pred_reconciled", "sum"))
        g_cal["rel"] = (g_cal["y"] - g_cal["yhat"]) / np.maximum(g_cal["yhat"], 10.0)
        g_oos = oos.groupby(cols_cal, as_index=False).agg(y=("y_true", "sum"), yhat=("pred_reconciled", "sum"))

        q_by_hb = {}
        for hb in ["1-6", "7-12", "13-19"]:
            sub = g_cal[g_cal["H_Bucket"] == hb]
            # Pool across horizons when bucket has <25 points (e.g. Global) for conformal stability
            if len(sub) < 25:
                sub = g_cal
            n = len(sub)
            rel_vals = sub["rel"].values
            # Center residuals around median so growth-regime shift doesn't starve the lower tail
            med_shift = float(np.median(rel_vals))
            abs_centered = np.abs(rel_vals - med_shift)
            q80_lvl = min(1.0, np.ceil((n + 1) * 0.80) / n)
            rad80 = float(np.quantile(abs_centered, q80_lvl))
            # 99% Tail-Risk Stress Envelope (symmetric on uncentered |rel| + finite-sample 1.08x margin)
            q99_lvl = min(1.0, np.ceil((n + 1) * 0.99) / n)
            rad99 = float(np.quantile(np.abs(rel_vals), q99_lvl)) * 1.08

            q_by_hb[hb] = {
                "q10": round(float(np.clip(-rad80, -0.60, -0.02)), 4),
                "q90": round(float(np.clip(rad80, 0.02, 0.70)), 4),
                "q01": round(float(np.clip(-rad99, -0.95, -0.05)), 4),
                "q99": round(float(np.clip(rad99, 0.05, 1.50)), 4),
            }

        # Evaluate strictly OUT-OF-SAMPLE on Origin 4
        lo80 = np.array([max(0.0, r.yhat * (1.0 + q_by_hb[r.H_Bucket]["q10"])) for r in g_oos.itertuples(index=False)])
        hi80 = np.array([max(r.yhat, r.yhat * (1.0 + q_by_hb[r.H_Bucket]["q90"])) for r in g_oos.itertuples(index=False)])
        lo99 = np.array([max(0.0, r.yhat * (1.0 + q_by_hb[r.H_Bucket]["q01"])) for r in g_oos.itertuples(index=False)])
        hi99 = np.array([max(r.yhat, r.yhat * (1.0 + q_by_hb[r.H_Bucket]["q99"])) for r in g_oos.itertuples(index=False)])
        y_o = g_oos["y"].values
        return q_by_hb, {
            "level": level_name,
            "oos_coverage_pct": round(float(((y_o >= lo80) & (y_o <= hi80)).mean()) * 100.0, 1),
            "oos_avg_width_pct_of_actual": round(float(np.sum(hi80 - lo80) / max(1.0, np.sum(y_o))) * 100.0, 1),
            "oos_p01_p99_coverage_pct": round(float(((y_o >= lo99) & (y_o <= hi99)).mean()) * 100.0, 1),
            "oos_p01_p99_width_pct": round(float(np.sum(hi99 - lo99) / max(1.0, np.sum(y_o))) * 100.0, 1),
            "n_oos_points": int(len(g_oos)),
        }

    agg_global, rep_global = fit_agg_level([], "Global Monthly Total")
    agg_family, rep_family = fit_agg_level(["Product_Family"], "Product Family Monthly")
    agg_region, rep_region = fit_agg_level(["Region"], "Region Monthly")
    agg_fam_reg, rep_fam_reg = fit_agg_level(["Product_Family", "Region"], "Family-Region Monthly")

    agg_intervals = {
        "calibration_method": "Centered split-conformal P10-P90 (target 80% OOS) + P01-P99 Stress Envelope (target 99% OOS), calibrated on Origins 1-3 and validated strictly out-of-sample on Origin 4",
        "global": agg_global,
        "family": agg_family,
        "region": agg_region,
        "family_region": agg_fam_reg,
    }
    coverage_report = {
        "target_coverage_pct": 80.0,
        "stress_envelope_target_pct": 99.0,
        "calibration_origins": ["2023-09", "2024-03", "2024-09"],
        "oos_validation_origin": "2025-03 (2025-04..2026-10)",
        "sku_conformal_scale_k": round(best_k, 3),
        "sku_level": {
            "level": "SKU-Region Monthly",
            "oos_coverage_pct": round(sku_oos_cov80, 1),
            "oos_avg_width_pct_of_actual": round(sku_oos_w80, 1),
            "oos_p01_p99_coverage_pct": round(sku_oos_cov99, 1),
            "oos_p01_p99_width_pct": round(sku_oos_w99, 1),
            "n_oos_points": int(len(oos)),
        },
        "aggregate_levels": [rep_global, rep_family, rep_region, rep_fam_reg],
    }
    return sku_q80_final, sku_q99_final, agg_intervals, coverage_report


def run_forecasting(raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> dict:
    train_df = pd.read_csv(out_dir / "train_enriched.csv.gz", low_memory=False)
    test_df = pd.read_csv(raw_dir / "test.csv")
    macro = pd.read_csv(out_dir / "macro_clean.csv")
    sku_master = pd.read_csv(raw_dir / "sku_master.csv")
    events = pd.read_csv(raw_dir / "events_calendar.csv")
    with open(out_dir / "macro_lead_lag.json") as f:
        macro_drivers = json.load(f)
    family_best_lags = macro_drivers.get("family_indicator_best_lags", {})

    ramp_up, ramp_down = calibrate_replacement_curves(train_df, sku_master)

    # 4 Rolling Origins (full 19-month horizon each!)
    origins = [
        ("2023-09", "2023-10", "2025-04"),
        ("2024-03", "2024-04", "2025-10"),
        ("2024-09", "2024-10", "2026-04"),
        ("2025-03", "2025-04", "2026-10"),
    ]
    bt_frames = []
    for orig, start_dt, end_dt in origins:
        val_actual = train_df[(train_df["Date"] >= start_dt) & (train_df["Date"] <= end_dt)].copy()
        val_keys = val_actual[
            ["Date", "Year", "Month", "Product_Family", "Business_Unit", "SKU_ID", "SKU_Description", "Customer_Name", "Sales_Channel", "Region"]
        ].copy()
        preds, _ = fit_and_predict_origin(
            train_df, val_keys, macro, sku_master, events, orig, ramp_up, ramp_down, family_best_lags
        )
        # Also evaluate the ACTUAL reconciled + rounded forecast in the backtest!
        hist_o = train_df[train_df["Date"] <= orig]
        rec_int, _ = apply_reconciliation_and_rounding(preds, hist_o, orig, "pred_ensemble_raw")
        preds["pred_reconciled"] = rec_int.astype(float)
        preds["y_true"] = val_actual["Unconstrained_Demand_For_Model"].values
        preds["y_shipped"] = val_actual["Units_Sold_Clean"].fillna(0.0).values
        preds["Origin"] = orig
        bt_frames.append(preds)

    bt = pd.concat(bt_frames, ignore_index=True)

    # Honest, code-accurate model names
    models = {
        "Seasonal Naive (Baseline 1)": "pred_snaive",
        "Moving Average 6M (Baseline 2)": "pred_ma6",
        "TSB Cycle-Aware (Intermittent)": "pred_tsb_cycle",
        "Ridge Regression (42M) + Heuristic Order Prob": "pred_ridge_ml",
        "Raw Ensemble (SBC + Lifecycle)": "pred_ensemble_raw",
        "Reconciled & Rounded Ensemble (Final Submission)": "pred_reconciled",
    }
    model_metrics = {}
    for m_name, col in models.items():
        w, b = calc_wape_bias(bt["y_true"].values, bt[col].values)
        model_metrics[m_name] = {"WAPE_Pct": w, "Bias_Pct": b}

    # Family, Region, Horizon metrics on the ACTUAL Reconciled & Rounded Final Submission model!
    by_family = {}
    for fam, g in bt.groupby("Product_Family"):
        w, b = calc_wape_bias(g["y_true"].values, g["pred_reconciled"].values)
        w_sn, _ = calc_wape_bias(g["y_true"].values, g["pred_snaive"].values)
        by_family[fam] = {"WAPE_Pct": w, "Bias_Pct": b, "Baseline_SN_WAPE_Pct": w_sn}

    by_region = {}
    for reg, g in bt.groupby("Region"):
        w, b = calc_wape_bias(g["y_true"].values, g["pred_reconciled"].values)
        w_sn, _ = calc_wape_bias(g["y_true"].values, g["pred_snaive"].values)
        by_region[reg] = {"WAPE_Pct": w, "Bias_Pct": b, "Baseline_SN_WAPE_Pct": w_sn}

    by_horizon = []
    for h, g in bt.groupby("Horizon"):
        w, b = calc_wape_bias(g["y_true"].values, g["pred_reconciled"].values)
        w_sn, _ = calc_wape_bias(g["y_true"].values, g["pred_snaive"].values)
        by_horizon.append({"Horizon": int(h), "WAPE_Pct": w, "Bias_Pct": b, "Baseline_SN_WAPE_Pct": w_sn})

    by_origin = []
    for orig, start_dt, end_dt in origins:
        g = bt[bt["Origin"] == orig]
        w_r, b_r = calc_wape_bias(g["y_true"].values, g["pred_reconciled"].values)
        w_sn, b_sn = calc_wape_bias(g["y_true"].values, g["pred_snaive"].values)
        fam_bias = {}
        for fam, fg in g.groupby("Product_Family"):
            wf, bf = calc_wape_bias(fg["y_true"].values, fg["pred_reconciled"].values)
            fam_bias[fam] = {"wape_pct": wf, "bias_pct": bf}
        by_origin.append({
            "origin": orig,
            "window": f"{start_dt}..{end_dt}",
            "reconciled_wape_pct": w_r,
            "reconciled_bias_pct": b_r,
            "snaive_wape_pct": w_sn,
            "snaive_bias_pct": b_sn,
            "by_family": fam_bias,
        })

    # Check backtest WAPE/Bias against actual shipped units (to verify stockout imputation bias)
    w_ship_rec, b_ship_rec = calc_wape_bias(bt["y_shipped"].values, bt["pred_reconciled"].values)
    w_ship_sn, b_ship_sn = calc_wape_bias(bt["y_shipped"].values, bt["pred_snaive"].values)
    shipped_actuals_check = {
        "reconciled_vs_shipped_wape_pct": w_ship_rec,
        "reconciled_vs_shipped_bias_pct": b_ship_rec,
        "snaive_vs_shipped_wape_pct": w_ship_sn,
        "snaive_vs_shipped_bias_pct": b_ship_sn,
    }

    # Aggregate Backtest Accuracy (Global Monthly Total & Family-Region Monthly)
    g_tot = bt.groupby(["Origin", "Date"], as_index=False).agg(
        y=("y_true", "sum"), rec=("pred_reconciled", "sum"), sn=("pred_snaive", "sum")
    )
    g_fr = bt.groupby(["Origin", "Date", "Product_Family", "Region"], as_index=False).agg(
        y=("y_true", "sum"), rec=("pred_reconciled", "sum"), sn=("pred_snaive", "sum")
    )
    w_gt_rec, b_gt_rec = calc_wape_bias(g_tot["y"].values, g_tot["rec"].values)
    w_gt_sn, b_gt_sn = calc_wape_bias(g_tot["y"].values, g_tot["sn"].values)
    w_fr_rec, b_fr_rec = calc_wape_bias(g_fr["y"].values, g_fr["rec"].values)
    w_fr_sn, b_fr_sn = calc_wape_bias(g_fr["y"].values, g_fr["sn"].values)
    aggregate_accuracy = {
        "global_monthly_total": {
            "reconciled_wape_pct": w_gt_rec, "reconciled_bias_pct": b_gt_rec,
            "snaive_wape_pct": w_gt_sn, "snaive_bias_pct": b_gt_sn,
        },
        "family_region_monthly": {
            "reconciled_wape_pct": w_fr_rec, "reconciled_bias_pct": b_fr_rec,
            "snaive_wape_pct": w_fr_sn, "snaive_bias_pct": b_fr_sn,
        },
    }

    # Out-of-sample conformal calibration at SKU level AND aggregate levels!
    sku_q_table, sku_q99_table, agg_intervals, coverage_report = calibrate_and_validate_intervals(bt)
    with open(out_dir / "agg_intervals.json", "w") as f:
        json.dump(agg_intervals, f, indent=2)

    # --- Final 19-Month Forecast on test.csv (2026-11 .. 2028-05) ---
    test_preds, final_elasticities = fit_and_predict_origin(
        train_df, test_df, macro, sku_master, events, "2026-10", ramp_up, ramp_down, family_best_lags
    )
    p50_int, recon_report = apply_reconciliation_and_rounding(test_preds, train_df, "2026-10", "pred_ensemble_raw")
    test_preds["Units_Sold_Forecast"] = p50_int
    test_preds["P50"] = p50_int

    # SKU-level calibrated P10 and P90
    h_buckets = pd.cut(test_preds["Horizon"], bins=[0, 6, 12, 20], labels=["1-6", "7-12", "13-19"]).astype(str)
    test_preds["H_Bucket"] = h_buckets
    p10_arr = np.zeros(len(test_preds), dtype=int)
    p90_arr = np.zeros(len(test_preds), dtype=int)
    for i, (sbc, hb, p50) in enumerate(zip(test_preds["SBC_Class"], h_buckets, p50_int)):
        if p50 == 0:
            p10_arr[i] = 0
            p90_arr[i] = 1 if sbc in ("Intermittent", "Lumpy") else 0
            continue
        q10, q90 = sku_q_table.get((sbc, hb), (-0.40, 0.45))
        p10_arr[i] = max(0, int(np.floor(p50 * (1.0 + q10))))
        p90_arr[i] = max(p50, int(np.ceil(p50 * (1.0 + q90))))

    p01_arr = np.zeros(len(test_preds), dtype=int)
    p99_arr = np.zeros(len(test_preds), dtype=int)
    for i, (sbc, hb, p50) in enumerate(zip(test_preds["SBC_Class"], h_buckets, p50_int)):
        if p50 == 0:
            p01_arr[i], p99_arr[i] = 0, (5 if sbc in ("Intermittent", "Lumpy") else 0)
            continue
        q01, q99 = sku_q99_table.get((sbc, hb), (-0.90, 1.40))
        p01_arr[i] = max(0, int(np.floor(p50 * (1.0 + q01))))
        p99_arr[i] = max(p90_arr[i], int(np.ceil(p50 * (1.0 + q99))))
    test_preds["P01"] = p01_arr
    test_preds["P10"] = p10_arr
    test_preds["P90"] = p90_arr
    test_preds["P99"] = p99_arr
    test_preds["Revenue_P50"] = np.round(test_preds["P50"] * test_preds["Unit_Price_Projected"], 2)
    test_preds["Revenue_P90"] = np.round(test_preds["P90"] * test_preds["Unit_Price_Projected"], 2)

    # Compute properly calibrated GLOBAL 19M P10/P90 (not summed SKU quantiles!)
    g_m = test_preds.groupby(["Date", "H_Bucket"], as_index=False).agg(p50=("P50", "sum"), rev50=("Revenue_P50", "sum"))
    glob_p10 = sum(max(0, int(round(r.p50 * (1.0 + agg_intervals["global"][r.H_Bucket]["q10"])))) for r in g_m.itertuples(index=False))
    glob_p90 = sum(max(int(r.p50), int(round(r.p50 * (1.0 + agg_intervals["global"][r.H_Bucket]["q90"])))) for r in g_m.itertuples(index=False))
    glob_rev_p90 = sum(r.rev50 * (1.0 + agg_intervals["global"][r.H_Bucket]["q90"]) for r in g_m.itertuples(index=False))

    # Save submission.csv
    sub = test_preds[["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]].copy()
    sub["Units_Sold_Forecast"] = sub["Units_Sold_Forecast"].astype(int)
    assert len(sub) == len(test_df)
    assert sub.isna().sum().sum() == 0
    assert (sub["Units_Sold_Forecast"] < 0).sum() == 0
    sub.to_csv(out_dir / "submission.csv", index=False)

    test_preds.to_csv(out_dir / "forecast_full.csv.gz", index=False, compression="gzip")

    backtest_summary = {
        "origins": [{"origin": o, "start": s, "end": e, "horizon_months": 19} for o, s, e in origins],
        "model_comparison": model_metrics,
        "by_family": by_family,
        "by_region": by_region,
        "by_horizon": by_horizon,
        "by_origin": by_origin,
        "aggregate_accuracy": aggregate_accuracy,
        "shipped_actuals_check": shipped_actuals_check,
        "interval_calibration": coverage_report,
        "what_if_elasticities": final_elasticities,
        "calibrated_ramp_up_curve": ramp_up,
        "calibrated_ramp_down_curve": ramp_down,
        "reconciliation": recon_report,
        "forecast_headline": {
            "total_rows": len(sub),
            "total_p10_units_global_calibrated": int(glob_p10),
            "total_p50_units": int(test_preds["P50"].sum()),
            "total_p90_units_global_calibrated": int(glob_p90),
            "total_p50_revenue_usd": round(float(test_preds["Revenue_P50"].sum()), 2),
            "total_p90_revenue_usd_global_calibrated": round(float(glob_rev_p90), 2),
            "sku_sum_p90_units_for_sku_level_buffer": int(test_preds["P90"].sum()),
        },
    }
    with open(out_dir / "backtest_metrics.json", "w") as f:
        json.dump(backtest_summary, f, indent=2)
    with open(out_dir / "reconciliation_check.json", "w") as f:
        json.dump(recon_report, f, indent=2)

    return backtest_summary


if __name__ == "__main__":
    summary = run_forecasting()
    print("=== 4-ORIGIN 19M BACKTEST LEADERBOARD ===")
    print(json.dumps(summary["model_comparison"], indent=2))
    print("\n=== OUT-OF-SAMPLE INTERVAL COVERAGE (TARGET 80%) ===")
    print(json.dumps(summary["interval_calibration"], indent=2))
    print("\n=== MODEL-DERIVED WHAT-IF ELASTICITIES ===")
    print(json.dumps(summary["what_if_elasticities"], indent=2))
    print("\n=== FORECAST HEADLINE (GLOBAL CALIBRATED P10/P90) ===")
    print(json.dumps(summary["forecast_headline"], indent=2))
