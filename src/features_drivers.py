import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUT_DIR, RAW_DIR


def detect_cycle_and_phase(vals: np.ndarray, month_indices: np.ndarray) -> tuple[int, int, float]:
    """Detect monthly(1), bi-monthly(2), or quarterly(3) ordering cadence on recent 24 months."""
    if len(vals) < 8:
        return 1, 0, 0.0
    nz = (vals > 0).astype(int)
    nz_rate = nz.mean()
    if nz_rate == 0:
        return 0, 0, 1.0
    if nz_rate >= 0.78:
        return 1, 0, float(nz_rate)

    best_p, best_phase, best_score = 1, 0, 0.0
    for p in (2, 3):
        for ph in range(p):
            on_mask = (month_indices % p) == ph
            off_mask = ~on_mask
            if on_mask.sum() < 3 or off_mask.sum() < 3:
                continue
            on_rate = nz[on_mask].mean()
            off_rate = nz[off_mask].mean()
            score = on_rate - off_rate
            if score > best_score:
                best_score = float(score)
                best_p = p
                best_phase = ph
    if best_score >= 0.55:
        return best_p, best_phase, round(best_score, 3)
    return 1, 0, round(best_score, 3)


def compute_intermittency_profiles(df: pd.DataFrame) -> pd.DataFrame:
    all_months = sorted(df["Date"].unique())
    m_idx = {m: i for i, m in enumerate(all_months)}
    df = df.copy()
    df["t_idx"] = df["Date"].map(m_idx)

    rows = []
    for (sid, reg), g in df.groupby(["SKU_ID", "Region"], sort=False):
        recent = g.tail(24)
        u_all = g["Unconstrained_Demand"].dropna().values
        u_rec = recent["Unconstrained_Demand"].dropna().values
        t_rec = recent.loc[recent["Unconstrained_Demand"].notna(), "t_idx"].values

        pos_rec = u_rec[u_rec > 0]
        n_obs = len(u_rec)
        n_pos = len(pos_rec)
        zero_pct = float((u_rec == 0).mean()) if n_obs > 0 else 1.0

        if n_pos == 0:
            adi = 99.0
            cv2 = 0.0
            sbc = "Inactive/Zero"
        else:
            adi = float(n_obs / n_pos)
            mean_p = float(pos_rec.mean())
            std_p = float(pos_rec.std(ddof=0)) if n_pos > 1 else 0.0
            cv2 = float((std_p / mean_p) ** 2) if mean_p > 0 else 0.0
            if adi < 1.32 and cv2 < 0.49:
                sbc = "Smooth"
            elif adi < 1.32 and cv2 >= 0.49:
                sbc = "Erratic"
            elif adi >= 1.32 and cv2 < 0.49:
                sbc = "Intermittent"
            else:
                sbc = "Lumpy"

        period, phase, cycle_strength = detect_cycle_and_phase(u_rec, t_rec)
        rows.append({
            "SKU_ID": sid,
            "Region": reg,
            "Product_Family": g["Product_Family"].iloc[0],
            "Customer_Name": g["Customer_Name"].iloc[-1],
            "Sales_Channel": g["Sales_Channel"].iloc[-1],
            "Active_Months": len(g),
            "Recent_24m_Mean": round(float(u_rec.mean()) if n_obs else 0.0, 2),
            "Recent_24m_Pos_Mean": round(float(pos_rec.mean()) if n_pos else 0.0, 2),
            "Zero_Month_Pct": round(zero_pct * 100.0, 1),
            "ADI": round(min(adi, 99.0), 3),
            "CV2": round(cv2, 3),
            "SBC_Class": sbc,
            "Detected_Cycle_Period": period,
            "Cycle_Phase": phase,
            "Cycle_Strength": cycle_strength,
            "Is_Lost_Account": bool(g["Is_Lost_Account"].iloc[-1]),
        })
    return pd.DataFrame(rows)


def analyze_macro_drivers_and_ev_shift(
    df: pd.DataFrame, macro: pd.DataFrame, sku_master: pd.DataFrame
) -> dict:
    indicators = [
        "ISM_Manufacturing_PMI",
        "Auto_Production_YoY_Pct",
        "Vehicle_Inventory_Days",
        "Raw_Material_Cost_Index",
        "New_Vehicle_Orders_Index",
        "Industrial_Util_Rate_Pct",
        "Fuel_Price_Index",
        "EV_Adoption_Index",
    ]
    lags = [0, 1, 2, 3, 4, 5, 6]

    # Monthly family unconstrained demand (exclude COVID shock 2020-03..2020-06 for cleaner structural lead/lag)
    fam_monthly = (
        df.groupby(["Date", "Product_Family"])["Unconstrained_Demand"]
        .sum()
        .unstack(fill_value=0.0)
        .sort_index()
    )
    m_df = macro.set_index("Date").reindex(fam_monthly.index)

    # Deseasonalize each family series via month-of-year ratio so seasonal cycles don't confound macro lag
    months = pd.Series([int(d[5:7]) for d in fam_monthly.index], index=fam_monthly.index)
    fam_deseas = pd.DataFrame(index=fam_monthly.index)
    for fam in fam_monthly.columns:
        s = fam_monthly[fam]
        m_means = s.groupby(months).transform("mean")
        fam_deseas[fam] = s / (m_means + 1e-6)

    heatmap = {}
    best_drivers = {}
    lag_stability = []

    early_mask = fam_deseas.index <= "2022-12"
    late_mask = fam_deseas.index >= "2023-01"

    for fam in sorted(fam_monthly.columns):
        y = fam_deseas[fam]
        # YoY growth of deseasonalized demand to remove secular trend when checking cyclical indicators
        y_yoy = y.pct_change(12)
        heatmap[fam] = {}
        best_ind = None
        best_lag = 0
        best_abs_r = -1.0
        best_r = 0.0

        for ind in indicators:
            x_raw = m_df[ind]
            # Use level for PMI/YoY/Util/EV_Adoption and YoY diff for indices
            x_series = x_raw if ind in ("ISM_Manufacturing_PMI", "Auto_Production_YoY_Pct", "Industrial_Util_Rate_Pct", "EV_Adoption_Index") else x_raw.pct_change(12)
            target = y if ind == "EV_Adoption_Index" else y_yoy

            lag_corrs = {}
            ind_best_lag = 0
            ind_best_r = 0.0
            for lag in lags:
                xs = x_series.shift(lag)
                valid = target.notna() & xs.notna() & np.isfinite(target) & np.isfinite(xs)
                r = float(np.corrcoef(target[valid], xs[valid])[0, 1]) if valid.sum() >= 18 else 0.0
                if np.isnan(r):
                    r = 0.0
                lag_corrs[str(lag)] = round(r, 3)
                if abs(r) > abs(ind_best_r):
                    ind_best_r = r
                    ind_best_lag = lag

            heatmap[fam][ind] = lag_corrs
            if abs(ind_best_r) > best_abs_r:
                best_abs_r = abs(ind_best_r)
                best_r = ind_best_r
                best_ind = ind
                best_lag = ind_best_lag

            # Early vs Late lag stability check
            def regime_best(rmask):
                br, bl = 0.0, 0
                for lag in lags:
                    xs = x_series.shift(lag)
                    v = rmask & target.notna() & xs.notna() & np.isfinite(target) & np.isfinite(xs)
                    if v.sum() >= 12:
                        rv = float(np.corrcoef(target[v], xs[v])[0, 1])
                        if not np.isnan(rv) and abs(rv) > abs(br):
                            br, bl = rv, lag
                return bl, round(br, 3)

            early_lag, early_r = regime_best(early_mask)
            late_lag, late_r = regime_best(late_mask)
            if abs(ind_best_r) >= 0.30:
                lag_stability.append({
                    "Product_Family": fam,
                    "Indicator": ind,
                    "Full_Best_Lag": ind_best_lag,
                    "Full_Corr": round(ind_best_r, 3),
                    "Early_2019_2022_Lag": early_lag,
                    "Early_Corr": early_r,
                    "Late_2023_2026_Lag": late_lag,
                    "Late_Corr": late_r,
                    "Lag_Shift_Months": late_lag - early_lag,
                    "Stable": bool(abs(late_lag - early_lag) <= 1 and (np.sign(early_r) == np.sign(late_r))),
                })

        best_drivers[fam] = {
            "leading_indicator": best_ind,
            "optimal_lag_months": best_lag,
            "correlation": round(best_r, 3),
        }

    # Expert: The Shift to Electric Vehicles (EV vs ICE vs Universal)
    df_pt = df.merge(sku_master[["SKU_ID", "Powertrain_Application"]], on="SKU_ID", how="left")
    pt_monthly = (
        df_pt.groupby(["Date", "Powertrain_Application"])["Unconstrained_Demand"]
        .sum()
        .unstack(fill_value=0.0)
        .sort_index()
    )
    pt_monthly["Total"] = pt_monthly.sum(axis=1)
    pt_monthly["EV_Share_Pct"] = np.round(100.0 * pt_monthly.get("EV", 0.0) / (pt_monthly["Total"] + 1e-6), 2)
    pt_monthly["ICE_Share_Pct"] = np.round(100.0 * pt_monthly.get("ICE", 0.0) / (pt_monthly["Total"] + 1e-6), 2)
    pt_monthly["EV_Adoption_Index"] = m_df["EV_Adoption_Index"].values

    ev_shift_records = []
    for dt, row in pt_monthly.iterrows():
        ev_shift_records.append({
            "Date": dt,
            "EV_Units": round(float(row.get("EV", 0.0)), 1),
            "ICE_Units": round(float(row.get("ICE", 0.0)), 1),
            "Universal_Units": round(float(row.get("Universal", 0.0)), 1),
            "EV_Share_Pct": float(row["EV_Share_Pct"]),
            "ICE_Share_Pct": float(row["ICE_Share_Pct"]),
            "EV_Adoption_Index": round(float(row["EV_Adoption_Index"]), 2),
        })

    # Annual EV vs ICE summary
    pt_annual = (
        df_pt.groupby(["Year", "Powertrain_Application"])["Unconstrained_Demand"]
        .sum()
        .unstack(fill_value=0.0)
    )
    ev_corr = float(np.corrcoef(pt_monthly["EV_Share_Pct"], pt_monthly["EV_Adoption_Index"])[0, 1])

    return {
        "indicators": indicators,
        "lags": lags,
        "family_best_drivers": best_drivers,
        "heatmap": heatmap,
        "lag_stability": lag_stability,
        "ev_shift_monthly": ev_shift_records,
        "ev_share_vs_adoption_corr": round(ev_corr, 3),
        "ev_share_2019_pct": float(pt_monthly.loc["2019-01":"2019-12", "EV_Share_Pct"].mean().round(2)),
        "ev_share_2026_pct": float(pt_monthly.loc["2026-01":"2026-10", "EV_Share_Pct"].mean().round(2)),
        "ice_share_2019_pct": float(pt_monthly.loc["2019-01":"2019-12", "ICE_Share_Pct"].mean().round(2)),
        "ice_share_2026_pct": float(pt_monthly.loc["2026-01":"2026-10", "ICE_Share_Pct"].mean().round(2)),
    }


def run_feature_driver_analysis(raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> dict:
    df = pd.read_csv(out_dir / "train_enriched.csv.gz", low_memory=False)
    macro = pd.read_csv(out_dir / "macro_clean.csv")
    sku_master = pd.read_csv(raw_dir / "sku_master.csv")

    interm_df = compute_intermittency_profiles(df)
    interm_df.to_csv(out_dir / "intermittency_profile.csv", index=False)

    driver_report = analyze_macro_drivers_and_ev_shift(df, macro, sku_master)
    with open(out_dir / "macro_lead_lag.json", "w") as f:
        json.dump(driver_report, f, indent=2)

    return {
        "sbc_distribution": interm_df["SBC_Class"].value_counts().to_dict(),
        "cycle_distribution": interm_df["Detected_Cycle_Period"].value_counts().to_dict(),
        "family_best_drivers": driver_report["family_best_drivers"],
        "ev_shift": {
            "ev_2019_pct": driver_report["ev_share_2019_pct"],
            "ev_2026_pct": driver_report["ev_share_2026_pct"],
            "ice_2019_pct": driver_report["ice_share_2019_pct"],
            "ice_2026_pct": driver_report["ice_share_2026_pct"],
            "corr_with_ev_index": driver_report["ev_share_vs_adoption_corr"],
        },
        "lag_stability_checked": len(driver_report["lag_stability"]),
    }


if __name__ == "__main__":
    res = run_feature_driver_analysis()
    print(json.dumps(res, indent=2))
