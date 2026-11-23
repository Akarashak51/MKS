import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUT_DIR, RAW_DIR


def detect_anomalies_and_unconstrain(raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> dict:
    df = pd.read_csv(out_dir / "train_clean.csv.gz", low_memory=False)
    stockouts = pd.read_csv(raw_dir / "inventory_stockouts.csv")
    sku_master = pd.read_csv(raw_dir / "sku_master.csv")
    events = pd.read_csv(raw_dir / "events_calendar.csv")

    df = df.sort_values(["SKU_ID", "Region", "Date"]).reset_index(drop=True)

    # Merge recorded stockouts
    st_cols = [
        "Date", "SKU_ID", "Region", "Requested_Units", "Carried_In_Units",
        "Shipped_Units", "Backordered_Units", "Lost_Units", "Fill_Rate_Pct",
        "Stockout_Days", "Reason",
    ]
    df = df.merge(stockouts[st_cols], on=["Date", "SKU_ID", "Region"], how="left")
    df["Stockout_Status"] = np.where(df["Requested_Units"].notna(), "recorded", "none")

    # Compute local robust rolling baseline per (SKU_ID, Region) using positive, non-recorded-stockout months
    df["Base_For_Med"] = np.where(
        (df["Units_Sold_Clean"] > 0) & (df["Stockout_Status"] == "none"),
        df["Units_Sold_Clean"],
        np.nan,
    )
    grp = df.groupby(["SKU_ID", "Region"], sort=False)
    df["Local_Med"] = grp["Base_For_Med"].transform(
        lambda s: s.rolling(window=9, min_periods=3, center=True).median()
    )
    df["Local_Med"] = grp["Local_Med"].transform(lambda s: s.bfill().ffill())
    df["Local_Std"] = grp["Base_For_Med"].transform(
        lambda s: s.rolling(window=9, min_periods=3, center=True).std()
    ).fillna(0.0)
    df["Prev_U"] = grp["Units_Sold_Clean"].shift(1)
    df["Next_U"] = grp["Units_Sold_Clean"].shift(-1)

    # Known disruption windows to avoid misclassifying macro/strike dips as isolated stockouts
    covid_mask = (
        ((df["Region"] == "Asia Pacific") & df["Date"].between("2020-02", "2020-03"))
        | ((df["Region"] == "Europe") & df["Date"].between("2020-03", "2020-05"))
        | ((df["Region"] == "India") & df["Date"].between("2020-04", "2020-06"))
        | ((df["Region"] == "Latin America") & df["Date"].between("2020-04", "2020-06"))
        | ((df["Region"] == "North America") & df["Date"].between("2020-04", "2020-06"))
    )
    nexon_strike_mask = (
        (df["Customer_Name"] == "Nexon Motor Group")
        & (df["Region"] == "Europe")
        & df["Date"].between("2023-10", "2023-11")
    )
    red_sea_mask = (
        (df["Sales_Channel"] == "Export")
        & df["Region"].isin(["Europe", "Middle East & Africa"])
        & df["Date"].between("2024-01", "2024-03")
    )
    holiday_shutdown_mask = (
        ((df["Region"] == "North America") & (df["Month"] == 7))
        | ((df["Region"] == "Europe") & (df["Month"] == 8))
        | ((df["Region"] == "Asia Pacific") & (df["Month"].isin([1, 2])))
    )
    ela_semi_window = (
        (df["Product_Family"] == "Electrical Assemblies")
        & df["Date"].between("2021-03", "2022-09")
    )

    # Detect UNRECORDED stockouts:
    # Case A: Electrical Assemblies during EV-019 semiconductor shortage not in inventory_stockouts.csv
    # Use pre/post shortage median per (SKU_ID, Region) as reference
    non_semi = df[~ela_semi_window & (df["Units_Sold_Clean"] > 0)]
    ela_ref = non_semi.groupby(["SKU_ID", "Region"])["Units_Sold_Clean"].median().rename("ELA_Ref_Med")
    df = df.merge(ela_ref, on=["SKU_ID", "Region"], how="left")
    unrec_semi = (
        ela_semi_window
        & (df["Stockout_Status"] == "none")
        & (df["Units_Sold_Clean"] > 0)
        & (df["ELA_Ref_Med"] >= 8)
        & (df["Units_Sold_Clean"] < 0.70 * df["ELA_Ref_Med"])
    )

    # Case B: Isolated sharp drop outside disruptions/holidays where both neighbors are healthy
    unrec_isolated = (
        (df["Stockout_Status"] == "none")
        & (~ela_semi_window)
        & (~covid_mask)
        & (~nexon_strike_mask)
        & (~red_sea_mask)
        & (~holiday_shutdown_mask)
        & (df["Units_Sold_Clean"] > 0)
        & (df["Local_Med"] >= 15)
        & (df["Units_Sold_Clean"] < 0.45 * df["Local_Med"])
        & (df["Prev_U"] >= 0.70 * df["Local_Med"])
        & (df["Next_U"] >= 0.70 * df["Local_Med"])
    )

    unrec_stockout = unrec_semi | unrec_isolated
    df.loc[unrec_stockout, "Stockout_Status"] = "detected_unrecorded"

    # Estimate Unconstrained Demand
    df["Unconstrained_Demand"] = df["Units_Sold_Clean"].copy()
    rec_mask = df["Stockout_Status"] == "recorded"
    df.loc[rec_mask, "Unconstrained_Demand"] = df.loc[rec_mask, "Requested_Units"]

    # For unrecorded semi stockouts, estimate from ELA_Ref_Med (or Local_Med)
    est_semi = np.round(np.maximum(df["ELA_Ref_Med"].fillna(df["Local_Med"]), df["Units_Sold_Clean"]))
    df.loc[unrec_semi, "Unconstrained_Demand"] = est_semi[unrec_semi]

    # For isolated unrecorded stockouts, estimate from average of neighbors / Local_Med
    est_iso = np.round(0.5 * (df["Prev_U"] + df["Next_U"]))
    df.loc[unrec_isolated, "Unconstrained_Demand"] = np.maximum(est_iso[unrec_isolated], df.loc[unrec_isolated, "Units_Sold_Clean"])

    # For missing panel rows (NaN Units_Sold_Clean), impute modeling demand only in Unconstrained_Demand_Imputed
    # while keeping Units_Sold_Clean as NaN!
    df["Unconstrained_Demand_For_Model"] = df["Unconstrained_Demand"].copy()
    miss_val_mask = df["Unconstrained_Demand_For_Model"].isna()
    df["Unconstrained_Demand_For_Model"] = (
        df.groupby(["SKU_ID", "Region"], sort=False)["Unconstrained_Demand_For_Model"]
        .transform(lambda s: s.interpolate(method="linear").bfill().ffill())
        .fillna(0.0)
    )

    df["Estimated_Lost_Units"] = np.where(
        df["Stockout_Status"] == "recorded",
        df["Lost_Units"].fillna(0.0),
        np.where(
            df["Stockout_Status"] == "detected_unrecorded",
            np.maximum(0.0, df["Unconstrained_Demand"] - df["Units_Sold_Clean"]),
            0.0,
        ),
    )

    # --- Build Anomaly Register (Real Events vs Errors) ---
    register_rows = []

    # 1. Data Errors already fixed in Workstream 1 (summarized + top examples & flagged rows)
    err_rows = df[df["Cleaning_Flag"].fillna("") != ""]
    for flag_val, grp_f in err_rows.groupby("Cleaning_Flag"):
        # Log top/representative or individual high-impact rows; keep register clean & complete
        is_missing_row = "missing_erp_row" in flag_val
        for r in grp_f.itertuples(index=False):
            if is_missing_row and r.Region == "Latin America" and "2023-03" <= r.Date <= "2023-05":
                atype = "ERP_Migration_Missing_Row"
                reason = "EV-033: Latin America ERP migration dropped transactions (distinct from 0)"
                action = "Keep as NaN in clean table; interpolate for model training"
            elif is_missing_row:
                atype = "Missing_Panel_Row"
                reason = "Missing SKU-Region-Month row between active dates (unknown, not zero)"
                action = "Keep as NaN in clean table; interpolate for model training"
            else:
                atype = f"Data_Error:{flag_val}"
                reason = f"ERP data quality defect ({flag_val})"
                action = "Fixed deterministically in Workstream 1 cleaning pipeline"
            register_rows.append({
                "Date": r.Date,
                "SKU_ID": r.SKU_ID,
                "Region": r.Region,
                "Product_Family": r.Product_Family,
                "Customer_Name": r.Customer_Name,
                "Category": "Data Error",
                "Anomaly_Type": atype,
                "Severity": "High" if ("100x" in flag_val or "sign_flip" in flag_val or "missing_erp" in flag_val) else "Medium",
                "Observed_Value": float(r.Units_Sold_Raw) if pd.notna(r.Units_Sold_Raw) else np.nan,
                "Adjusted_Value": float(r.Units_Sold_Clean) if pd.notna(r.Units_Sold_Clean) else np.nan,
                "Reason": reason,
                "Recommended_Action": action,
            })

    # 2. Recorded & Unrecorded Stockouts
    for r in df[df["Stockout_Status"] == "detected_unrecorded"].itertuples(index=False):
        is_semi = (r.Product_Family == "Electrical Assemblies") and ("2021-03" <= r.Date <= "2022-09")
        register_rows.append({
            "Date": r.Date,
            "SKU_ID": r.SKU_ID,
            "Region": r.Region,
            "Product_Family": r.Product_Family,
            "Customer_Name": r.Customer_Name,
            "Category": "Real Event (Supply Constraint)",
            "Anomaly_Type": "Unrecorded_Semiconductor_Stockout" if is_semi else "Unrecorded_Isolated_Stockout",
            "Severity": "High",
            "Observed_Value": float(r.Units_Sold_Clean),
            "Adjusted_Value": float(r.Unconstrained_Demand),
            "Reason": "EV-019 semiconductor shortage unlogged in inventory_stockouts.csv" if is_semi else "Sharp >55% single-month drop vs healthy neighbor months",
            "Recommended_Action": "Replace constrained shipment with estimated unconstrained demand for forecasting",
        })

    # 3. Early Buying / Forward-Buying before Jan 2023 9% price rise (EV-030 -> Dec 2022 spike)
    dec22 = df[df["Date"] == "2022-12"].set_index(["SKU_ID", "Region"])
    oct_nov22 = df[df["Date"].isin(["2022-10", "2022-11"]) & (df["Units_Sold_Clean"] > 0)].groupby(["SKU_ID", "Region"])["Units_Sold_Clean"].mean()
    jan23 = df[df["Date"] == "2023-01"].set_index(["SKU_ID", "Region"])["Units_Sold_Clean"]
    fb_count = 0
    for key, row in dec22.iterrows():
        base = oct_nov22.get(key, np.nan)
        j23 = jan23.get(key, np.nan)
        u_dec = row["Unconstrained_Demand"]
        if pd.notna(base) and base >= 10 and pd.notna(u_dec) and u_dec > 1.35 * base:
            fb_count += 1
            register_rows.append({
                "Date": "2022-12",
                "SKU_ID": key[0],
                "Region": key[1],
                "Product_Family": row["Product_Family"],
                "Customer_Name": row["Customer_Name"],
                "Category": "Real Event (Commercial)",
                "Anomaly_Type": "Forward_Buying_Pre_Price_Hike",
                "Severity": "Medium",
                "Observed_Value": float(u_dec),
                "Adjusted_Value": float(round(base, 1)),
                "Reason": f"EV-030: Pre-buy surge before 9% Jan-2023 price hike (Dec={u_dec:.0f} vs Oct-Nov avg={base:.1f}, Jan-23={j23 if pd.notna(j23) else -1:.0f})",
                "Recommended_Action": "Model via pre_price_hike_pct event feature; do not extrapolate as permanent trend",
            })

    # 4. One-off Demand Spikes (real events, non-forward-buying)
    spike_mask = (
        (df["Stockout_Status"] == "none")
        & (df["Date"] != "2022-12")
        & (df["Local_Med"] >= 10)
        & (df["Units_Sold_Clean"] > np.maximum(2.3 * df["Local_Med"], df["Local_Med"] + 3.5 * df["Local_Std"]))
    )
    for r in df[spike_mask].itertuples(index=False):
        register_rows.append({
            "Date": r.Date,
            "SKU_ID": r.SKU_ID,
            "Region": r.Region,
            "Product_Family": r.Product_Family,
            "Customer_Name": r.Customer_Name,
            "Category": "Real Event (Demand Shock)",
            "Anomaly_Type": "One_Off_Demand_Spike",
            "Severity": "Medium",
            "Observed_Value": float(r.Units_Sold_Clean),
            "Adjusted_Value": float(r.Local_Med),
            "Reason": f"Unusual single-month surge ({r.Units_Sold_Clean:.0f} vs local median {r.Local_Med:.1f})",
            "Recommended_Action": "Keep in history; dampen weight in forecasting so baseline is not inflated",
        })

    # 5. Structural Breaks: Helios Auto Parts India (EV-052), Nexon Strike (EV-038), Red Sea (EV-039)
    helios_before = df[(df["Customer_Name"] == "Helios Auto Parts") & (df["Region"] == "India") & (df["Date"].between("2024-09", "2025-08"))].groupby("SKU_ID")["Unconstrained_Demand"].mean()
    helios_after = df[(df["Customer_Name"] == "Helios Auto Parts") & (df["Region"] == "India") & (df["Date"] >= "2025-09")].groupby("SKU_ID")["Unconstrained_Demand"].mean()
    helios_breaks = 0
    for sid, after_m in helios_after.items():
        bef_m = helios_before.get(sid, np.nan)
        if pd.notna(bef_m) and bef_m > 5 and after_m > 1.25 * bef_m:
            helios_breaks += 1
            pf = df.loc[df["SKU_ID"] == sid, "Product_Family"].iloc[0]
            register_rows.append({
                "Date": "2025-09",
                "SKU_ID": sid,
                "Region": "India",
                "Product_Family": pf,
                "Customer_Name": "Helios Auto Parts",
                "Category": "Real Event (Structural Break)",
                "Anomaly_Type": "Permanent_Capacity_Step_Up",
                "Severity": "High",
                "Observed_Value": float(round(after_m, 1)),
                "Adjusted_Value": float(round(after_m, 1)),
                "Reason": f"EV-052: Helios Auto Parts 2nd India line raised mean demand from {bef_m:.1f} to {after_m:.1f} (+{(after_m/bef_m - 1)*100:.0f}%)",
                "Recommended_Action": "Model as permanent structural level shift in forecast window",
            })

    # 6. Expert: Customer Switches per (SKU_ID, Region)
    cust_switches = []
    for (sid, reg), g in df.groupby(["SKU_ID", "Region"], sort=False):
        c_vals = g["Customer_Name"].values
        dates = g["Date"].values
        changes = np.where(c_vals[1:] != c_vals[:-1])[0]
        for ch_idx in changes:
            old_c, new_c = c_vals[ch_idx], c_vals[ch_idx + 1]
            dt = dates[ch_idx + 1]
            cust_switches.append((dt, sid, reg, old_c, new_c))
            register_rows.append({
                "Date": dt,
                "SKU_ID": sid,
                "Region": reg,
                "Product_Family": g["Product_Family"].iloc[0],
                "Customer_Name": f"{old_c} -> {new_c}",
                "Category": "Real Event (Account Change)",
                "Anomaly_Type": "Customer_Switch",
                "Severity": "Medium",
                "Observed_Value": float(g["Unconstrained_Demand"].iloc[ch_idx + 1]) if pd.notna(g["Unconstrained_Demand"].iloc[ch_idx + 1]) else np.nan,
                "Adjusted_Value": np.nan,
                "Reason": f"Account transitioned from {old_c} to {new_c} in {reg}",
                "Recommended_Action": "Use post-switch level/cadence for forecast; include Customer_Name feature",
            })

    # 7. Expert: Ordering-Cycle Changes (ADI shift between first half and last 24 months)
    cycle_changes = 0
    for (sid, reg), g in df.groupby(["SKU_ID", "Region"], sort=False):
        if len(g) < 36:
            continue
        early = g.iloc[:-24]["Units_Sold_Clean"].dropna()
        late = g.iloc[-24:]["Units_Sold_Clean"].dropna()
        if len(early) < 12 or len(late) < 12:
            continue
        nz_early = (early > 0).mean()
        nz_late = (late > 0).mean()
        if nz_early > 0.05 and nz_late > 0.05:
            adi_early = 1.0 / nz_early
            adi_late = 1.0 / nz_late
            # Significant cadence change (e.g. monthly ~1.0 -> bi-monthly ~2.0 or quarterly ~3.0)
            if abs(adi_late - adi_early) >= 0.75:
                cycle_changes += 1
                def label_cadence(adi):
                    if adi < 1.35: return "Monthly"
                    if adi < 2.4: return "Bi-Monthly"
                    if adi < 3.6: return "Quarterly"
                    return "Sporadic"
                c_e, c_l = label_cadence(adi_early), label_cadence(adi_late)
                if c_e != c_l:
                    register_rows.append({
                        "Date": g["Date"].iloc[-24],
                        "SKU_ID": sid,
                        "Region": reg,
                        "Product_Family": g["Product_Family"].iloc[0],
                        "Customer_Name": g["Customer_Name"].iloc[-1],
                        "Category": "Real Event (Behavioral Shift)",
                        "Anomaly_Type": "Ordering_Cycle_Change",
                        "Severity": "Medium",
                        "Observed_Value": round(float(adi_late), 2),
                        "Adjusted_Value": round(float(adi_early), 2),
                        "Reason": f"Order cadence shifted from {c_e} (ADI={adi_early:.2f}) to {c_l} (ADI={adi_late:.2f})",
                        "Recommended_Action": "Use recent 18-24m cadence for intermittent/cycle forecasting (TSB/cycle-aware)",
                    })

    # 8. Expert: Lost Accounts (active SKU, >=6 prior positive months, then >=6 consecutive zeros through 2026-10)
    active_skus = set(sku_master.loc[sku_master["Lifecycle_Status"] == "Active", "SKU_ID"])
    lost_accounts = 0
    lost_account_keys = set()
    for (sid, reg), g in df.groupby(["SKU_ID", "Region"], sort=False):
        if sid not in active_skus:
            continue
        if g["Date"].iloc[-1] != "2026-10":
            continue
        u = g["Units_Sold_Clean"].fillna(0).values
        if (u > 0).sum() >= 6 and len(u) >= 12 and np.all(u[-6:] == 0):
            # Find first month of trailing zero streak
            last_pos = np.where(u > 0)[0][-1]
            streak_len = len(u) - 1 - last_pos
            if streak_len >= 6:
                lost_accounts += 1
                lost_account_keys.add((sid, reg))
                register_rows.append({
                    "Date": g["Date"].iloc[last_pos + 1],
                    "SKU_ID": sid,
                    "Region": reg,
                    "Product_Family": g["Product_Family"].iloc[0],
                    "Customer_Name": g["Customer_Name"].iloc[-1],
                    "Category": "Real Event (Account Change)",
                    "Anomaly_Type": "Lost_Account",
                    "Severity": "High",
                    "Observed_Value": 0.0,
                    "Adjusted_Value": 0.0,
                    "Reason": f"Active SKU went to 0 demand for {streak_len} consecutive months through 2026-10 (prior avg={u[:last_pos+1][u[:last_pos+1]>0].mean():.1f})",
                    "Recommended_Action": "Forecast near-zero / zero demand unless account win-back is confirmed",
                })

    # 9. Expert: Master Data vs Transactions contradictions
    df_m = df.merge(sku_master[["SKU_ID", "Launch_Date", "EOL_Date", "Lifecycle_Status"]], on="SKU_ID", how="left")
    pre_launch = df_m[(df_m["Units_Sold_Clean"] > 0) & (df_m["Date"] < df_m["Launch_Date"])]
    for r in pre_launch.itertuples(index=False):
        register_rows.append({
            "Date": r.Date,
            "SKU_ID": r.SKU_ID,
            "Region": r.Region,
            "Product_Family": r.Product_Family,
            "Customer_Name": r.Customer_Name,
            "Category": "Master Data Contradiction",
            "Anomaly_Type": "Sales_Before_Master_Launch_Date",
            "Severity": "High",
            "Observed_Value": float(r.Units_Sold_Clean),
            "Adjusted_Value": float(r.Units_Sold_Clean),
            "Reason": f"Transaction in {r.Date} precedes sku_master Launch_Date ({r.Launch_Date}); actual launch was earlier",
            "Recommended_Action": "Retain verified transactions; flag sku_master Launch_Date for MDM correction",
        })

    # Mark row-level flags on df
    df["Is_Forward_Buy_2022_12"] = (df["Date"] == "2022-12") & (
        df["Unconstrained_Demand"] > 1.35 * df["Local_Med"].fillna(1e9)
    )
    df["Is_One_Off_Spike"] = spike_mask
    df["Is_Lost_Account"] = [
        (s, r) in lost_account_keys for s, r in zip(df["SKU_ID"], df["Region"])
    ]
    df["Is_Covid_Disruption"] = covid_mask
    df["Is_Nexon_Strike"] = nexon_strike_mask
    df["Is_Red_Sea"] = red_sea_mask

    # Save outputs
    reg_df = pd.DataFrame(register_rows)
    reg_df = reg_df.sort_values(["Severity", "Date", "SKU_ID"]).reset_index(drop=True)
    reg_df.to_csv(out_dir / "anomaly_register.csv", index=False)

    df.to_csv(out_dir / "train_enriched.csv.gz", index=False)

    # Stockout summary
    stockout_summary = {
        "recorded_stockout_rows": int(rec_mask.sum()),
        "detected_unrecorded_stockout_rows": int(unrec_stockout.sum()),
        "detected_unrecorded_semi_rows": int(unrec_semi.sum()),
        "detected_unrecorded_isolated_rows": int(unrec_isolated.sum()),
        "total_recorded_lost_units": float(df.loc[rec_mask, "Lost_Units"].sum()),
        "total_detected_unrecorded_lost_units": float(df.loc[unrec_stockout, "Estimated_Lost_Units"].sum()),
        "total_shipped_units_clean": float(df["Units_Sold_Clean"].sum()),
        "total_unconstrained_demand": float(df["Unconstrained_Demand"].sum()),
        "total_uplift_units": float(df["Unconstrained_Demand"].sum() - df["Units_Sold_Clean"].sum()),
    }

    summary = {
        "total_register_entries": len(reg_df),
        "by_category": reg_df["Category"].value_counts().to_dict(),
        "by_type": reg_df["Anomaly_Type"].value_counts().head(25).to_dict(),
        "forward_buying_2022_12_series": fb_count,
        "helios_india_structural_breaks": helios_breaks,
        "customer_switches": len(cust_switches),
        "ordering_cycle_changes": cycle_changes,
        "lost_accounts": lost_accounts,
        "master_data_contradictions": len(pre_launch),
        "stockout_summary": stockout_summary,
    }
    with open(out_dir / "anomaly_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    s = detect_anomalies_and_unconstrain()
    print(json.dumps(s, indent=2))
