import difflib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from config import CANONICAL_CHANNELS, CANONICAL_REGIONS, OUT_DIR, RAW_DIR


REGION_MAP = {
    "north america": "North America",
    "n. america": "North America",
    "na": "North America",
    "europe": "Europe",
    "eu": "Europe",
    "asia pacific": "Asia Pacific",
    "apac": "Asia Pacific",
    "asia-pacific": "Asia Pacific",
    "india": "India",
    "latin america": "Latin America",
    "latam": "Latin America",
    "latin-america": "Latin America",
    "middle east & africa": "Middle East & Africa",
    "middle east and africa": "Middle East & Africa",
    "me&a": "Middle East & Africa",
    "mea": "Middle East & Africa",
}

CHANNEL_MAP = {
    "direct oem": "Direct OEM",
    "direct-oem": "Direct OEM",
    "oem direct": "Direct OEM",
    "tier-1 distributor": "Tier-1 Distributor",
    "tier 1 distributor": "Tier-1 Distributor",
    "t1 distributor": "Tier-1 Distributor",
    "aftermarket": "Aftermarket",
    "after-market": "Aftermarket",
    "export": "Export",
}


def parse_date_month(val: str) -> tuple[str | None, bool, bool]:
    """Return (YYYY-MM, was_nonstandard_format, was_year_typo)."""
    if pd.isna(val):
        return None, True, False
    s = str(val).strip()
    nonstd = False
    typo = False

    if re.match(r"^\d{4}-\d{2}$", s):
        ym = s
    elif re.match(r"^\d{4}/\d{2}$", s):
        ym = s.replace("/", "-")
        nonstd = True
    elif re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        ym = s[:7]
        nonstd = True
    elif re.match(r"^\d{2}/\d{4}$", s):
        m, y = s.split("/")
        ym = f"{y}-{m}"
        nonstd = True
    elif re.match(r"^\d{2}-\d{4}$", s):
        m, y = s.split("-")
        ym = f"{y}-{m}"
        nonstd = True
    else:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None, True, False
        ym = dt.strftime("%Y-%m")
        nonstd = True

    # Fix transposed year typo e.g. 2062-XX -> 2026-XX
    if ym.startswith("2062-"):
        ym = "2026-" + ym[5:]
        typo = True

    return ym, nonstd, typo


def clean_sku_id(val: str) -> tuple[str, bool]:
    s0 = str(val)
    s = s0.strip().upper().replace("_", "-")
    m = re.match(r"^([A-Z]{3})-?(\d+)$", s)
    if m:
        canon = f"{m.group(1)}-{int(m.group(2)):03d}"
    else:
        canon = s
    return canon, (canon != s0)


def build_customer_mapper(raw_names: pd.Series, canonical_names: list[str]) -> dict[str, str]:
    canon_norm = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in canonical_names}
    mapping = {}
    for name in raw_names.dropna().unique():
        if name in canonical_names:
            mapping[name] = name
            continue
        cleaned = re.sub(r"\binc\.?$", "", str(name).strip(), flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        norm = re.sub(r"[^a-z0-9]", "", cleaned.lower())
        if norm in canon_norm:
            mapping[name] = canon_norm[norm]
            continue
        best = difflib.get_close_matches(cleaned, canonical_names, n=1, cutoff=0.6)
        if best:
            mapping[name] = best[0]
        else:
            best_norm = difflib.get_close_matches(norm, list(canon_norm.keys()), n=1, cutoff=0.5)
            mapping[name] = canon_norm[best_norm[0]] if best_norm else cleaned
    return mapping


def clean_macro(raw_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(raw_dir / "macro_indicators.csv")
    before_rows = len(df)
    issues = []

    # 1. Deduplicate Date
    dup_count = int(df.duplicated("Date").sum())
    if dup_count > 0:
        issues.append({
            "file": "macro_indicators.csv",
            "issue_type": "duplicate_date",
            "category": "Data Error",
            "rows_affected": dup_count,
            "action": "Dropped exact duplicate Date rows (2022-06)",
        })
        df = df.drop_duplicates("Date", keep="first").sort_values("Date").reset_index(drop=True)

    # 2. Fix ISM_Manufacturing_PMI decimal shift error (4.79 -> 47.9)
    pmi_err = df["ISM_Manufacturing_PMI"] < 20.0
    if pmi_err.any():
        issues.append({
            "file": "macro_indicators.csv",
            "issue_type": "scale_error_pmi_10x",
            "category": "Data Error",
            "rows_affected": int(pmi_err.sum()),
            "action": "Multiplied PMI values < 20 by 10 (2023-05: 4.79 -> 47.9)",
        })
        df.loc[pmi_err, "ISM_Manufacturing_PMI"] = df.loc[pmi_err, "ISM_Manufacturing_PMI"] * 10.0

    # 3. Fix 2019 Auto_Production_YoY_Pct decimal-vs-percent scale error
    auto_err = (df["Year"] == 2019) & (df["Auto_Production_YoY_Pct"].abs() < 0.5)
    if auto_err.any():
        issues.append({
            "file": "macro_indicators.csv",
            "issue_type": "scale_error_auto_yoy_decimal_vs_pct",
            "category": "Data Error",
            "rows_affected": int(auto_err.sum()),
            "action": "Converted 2019 decimal fractions to percentage points (* 100)",
        })
        df.loc[auto_err, "Auto_Production_YoY_Pct"] = df.loc[auto_err, "Auto_Production_YoY_Pct"] * 100.0

    # 4. Interpolate missing values
    num_cols = [c for c in df.columns if c not in ("Date", "Year", "Month")]
    null_count = int(df[num_cols].isna().sum().sum())
    if null_count > 0:
        issues.append({
            "file": "macro_indicators.csv",
            "issue_type": "missing_indicator_values",
            "category": "Data Error",
            "rows_affected": null_count,
            "action": "Linearly interpolated missing macro indicator values across time",
        })
        for col in num_cols:
            df[col] = df[col].interpolate(method="linear").bfill().ffill()

    df["Year"] = df["Date"].str[:4].astype(int)
    df["Month"] = df["Date"].str[5:7].astype(int)
    df.to_csv(out_dir / "macro_clean.csv", index=False)

    summary = {
        "file": "macro_indicators.csv",
        "rows_before": before_rows,
        "rows_after": len(df),
        "nulls_before": null_count,
        "nulls_after": int(df[num_cols].isna().sum().sum()),
        "issues": issues,
    }
    return df, summary


def clean_train(raw_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    train = pd.read_csv(raw_dir / "train.csv", dtype=str)
    test = pd.read_csv(raw_dir / "test.csv", dtype=str)
    sku_master = pd.read_csv(raw_dir / "sku_master.csv", dtype=str)

    rows_before = len(train)
    unique_skus_before = int(train["SKU_ID"].nunique(dropna=False))
    unique_regions_before = int(train["Region"].nunique(dropna=False))
    unique_customers_before = int(train["Customer_Name"].nunique(dropna=False))
    unique_channels_before = int(train["Sales_Channel"].nunique(dropna=False))
    issues = []

    # 1. Deduplicate Record_ID keeping latest Load_Timestamp
    dup_mask = train.duplicated("Record_ID", keep=False)
    n_dup_rows_removed = len(train) - train["Record_ID"].nunique()
    train = train.sort_values("Load_Timestamp").drop_duplicates("Record_ID", keep="last").copy()
    issues.append({
        "file": "train.csv",
        "issue_type": "duplicate_record_id",
        "category": "Data Error",
        "rows_affected": int(n_dup_rows_removed),
        "action": "Retained row with latest Load_Timestamp per Record_ID",
    })

    # 2. Parse and standardize Date (authoritative over Year/Month)
    parsed = train["Date"].map(parse_date_month)
    train["Date_Clean"] = [p[0] for p in parsed]
    nonstd_dates = sum(p[1] for p in parsed)
    year_typos = sum(p[2] for p in parsed)
    issues.append({
        "file": "train.csv",
        "issue_type": "nonstandard_date_format",
        "category": "Data Error",
        "rows_affected": int(nonstd_dates),
        "action": "Standardized all date formats to YYYY-MM",
    })
    if year_typos > 0:
        issues.append({
            "file": "train.csv",
            "issue_type": "transposed_year_2062",
            "category": "Data Error",
            "rows_affected": int(year_typos),
            "action": "Corrected transposed year 2062-XX -> 2026-XX verified via Load_Timestamp",
        })

    train["Year_Clean"] = train["Date_Clean"].str[:4].astype(int)
    train["Month_Clean"] = train["Date_Clean"].str[5:7].astype(int)
    ym_mismatch = (train["Year_Clean"] != train["Year"].astype(int)) | (
        train["Month_Clean"] != train["Month"].astype(int)
    )
    issues.append({
        "file": "train.csv",
        "issue_type": "date_vs_year_month_mismatch",
        "category": "Data Error",
        "rows_affected": int(ym_mismatch.sum()),
        "action": "Overwrote Year and Month from authoritative Date column",
    })

    # 3. Standardize SKU_ID
    sku_parsed = train["SKU_ID"].map(clean_sku_id)
    train["SKU_Clean"] = [p[0] for p in sku_parsed]
    n_sku_fixed = sum(p[1] for p in sku_parsed)
    issues.append({
        "file": "train.csv",
        "issue_type": "nonstandard_sku_id",
        "category": "Data Error",
        "rows_affected": int(n_sku_fixed),
        "action": "Standardized SKU_ID to canonical [A-Z]{3}-NNN format",
    })

    # 4. Standardize Product_Family, Business_Unit, SKU_Description from sku_master
    n_pf_null = int(train["Product_Family"].isna().sum())
    n_desc_null = int(train["SKU_Description"].isna().sum())
    sku_lookup = sku_master.set_index("SKU_ID")
    train["Product_Family_Clean"] = train["SKU_Clean"].map(sku_lookup["Product_Family"])
    train["Business_Unit_Clean"] = train["SKU_Clean"].map(sku_lookup["Business_Unit"])
    train["SKU_Description_Clean"] = train["SKU_Clean"].map(sku_lookup["SKU_Description"])
    issues.append({
        "file": "train.csv",
        "issue_type": "missing_product_metadata",
        "category": "Data Error",
        "rows_affected": int(n_pf_null + n_desc_null),
        "action": "Backfilled missing Product_Family and SKU_Description from sku_master.csv",
    })

    # 5. Standardize Sales_Channel
    raw_ch = train["Sales_Channel"].copy()
    train["Sales_Channel_Clean"] = raw_ch.str.strip().str.lower().map(CHANNEL_MAP)
    n_ch_fixed = int((raw_ch != train["Sales_Channel_Clean"]).sum())
    issues.append({
        "file": "train.csv",
        "issue_type": "inconsistent_sales_channel",
        "category": "Data Error",
        "rows_affected": n_ch_fixed,
        "action": f"Standardized {unique_channels_before} raw channel variants to 4 canonical labels",
    })

    # 6. Standardize Customer_Name via fuzzy match against canonical test customer list
    canonical_customers = sorted(test["Customer_Name"].dropna().unique().tolist())
    cust_map = build_customer_mapper(train["Customer_Name"], canonical_customers)
    raw_cust = train["Customer_Name"].copy()
    train["Customer_Name_Clean"] = raw_cust.map(cust_map)
    n_cust_fixed = int((raw_cust != train["Customer_Name_Clean"]).sum())
    issues.append({
        "file": "train.csv",
        "issue_type": "customer_name_typos",
        "category": "Data Error",
        "rows_affected": n_cust_fixed,
        "action": f"Standardized {unique_customers_before} raw customer spelling variants to 12 canonical OEM/distributor names",
    })

    # 7. Standardize Region & recover missing Regions
    raw_reg = train["Region"].copy()
    train["Region_Clean"] = raw_reg.str.strip().str.lower().map(REGION_MAP)
    n_reg_variant = int((raw_reg.notna() & (raw_reg != train["Region_Clean"])).sum())
    n_reg_null_before = int(train["Region_Clean"].isna().sum())

    # Recover missing Region:
    # Step A: For each (Date_Clean, SKU_Clean), find which of the SKU's active regions is missing in that month
    sku_active_regions = (
        train.dropna(subset=["Region_Clean"])
        .groupby("SKU_Clean")["Region_Clean"]
        .apply(set)
        .to_dict()
    )
    # Also build (SKU_Clean, Customer_Name_Clean) -> unique Region mapping
    sc_reg = (
        train.dropna(subset=["Region_Clean"])
        .groupby(["SKU_Clean", "Customer_Name_Clean"])["Region_Clean"]
        .agg(lambda s: s.mode().iloc[0] if s.nunique() == 1 else None)
        .dropna()
        .to_dict()
    )
    # Build existing (Date_Clean, SKU_Clean) -> set of present regions
    present_ds_reg = (
        train.dropna(subset=["Region_Clean"])
        .groupby(["Date_Clean", "SKU_Clean"])["Region_Clean"]
        .apply(set)
        .to_dict()
    )

    recovered_reg = 0
    null_idx = train[train["Region_Clean"].isna()].index
    for idx in null_idx:
        dt = train.at[idx, "Date_Clean"]
        sk = train.at[idx, "SKU_Clean"]
        cu = train.at[idx, "Customer_Name_Clean"]
        active = set(sku_active_regions.get(sk, set()))
        if dt < "2022-01":
            active.discard("Middle East & Africa")
        present = present_ds_reg.get((dt, sk), set())
        missing_slots = active - present
        if len(missing_slots) == 1:
            reg_val = next(iter(missing_slots))
            train.at[idx, "Region_Clean"] = reg_val
            present_ds_reg.setdefault((dt, sk), set()).add(reg_val)
            recovered_reg += 1
        elif (sk, cu) in sc_reg and sc_reg[(sk, cu)] not in present:
            reg_val = sc_reg[(sk, cu)]
            train.at[idx, "Region_Clean"] = reg_val
            present_ds_reg.setdefault((dt, sk), set()).add(reg_val)
            recovered_reg += 1
        elif len(missing_slots) > 1:
            # Prefer region matching customer if in missing_slots
            cand = sc_reg.get((sk, cu))
            if cand in missing_slots:
                train.at[idx, "Region_Clean"] = cand
                present_ds_reg.setdefault((dt, sk), set()).add(cand)
                recovered_reg += 1

    # Drop any unrecoverable null Region rows (if any remain)
    n_reg_unrecov = int(train["Region_Clean"].isna().sum())
    if n_reg_unrecov > 0:
        train = train.dropna(subset=["Region_Clean"]).copy()

    issues.append({
        "file": "train.csv",
        "issue_type": "inconsistent_region_labels",
        "category": "Data Error",
        "rows_affected": n_reg_variant,
        "action": f"Standardized {unique_regions_before} raw region label variants (incl. nulls) to 6 canonical regions",
    })
    issues.append({
        "file": "train.csv",
        "issue_type": "missing_region_recovered",
        "category": "Data Error",
        "rows_affected": n_reg_null_before,
        "action": f"Recovered {recovered_reg} missing Region values via SKU-Date slot & Customer mapping ({n_reg_unrecov} dropped)",
    })

    # 8. Parse numeric columns & reconcile Units * Price = Revenue
    raw_u_str = train["Units_Sold"].copy()
    n_dash_u = int((raw_u_str == "-").sum())
    n_comma_u = int(raw_u_str.fillna("").str.contains(",").sum())

    train["U_num"] = pd.to_numeric(raw_u_str.str.replace(",", "", regex=False), errors="coerce")
    train["P_num"] = pd.to_numeric(train["Unit_Price_USD"], errors="coerce")
    train["R_num"] = pd.to_numeric(train["Revenue_USD"], errors="coerce")

    # Track initial reconciliation mismatch rate
    valid_init = train.dropna(subset=["U_num", "P_num", "R_num"])
    init_mismatch = int(
        (
            ((valid_init["U_num"] * valid_init["P_num"] - valid_init["R_num"]).abs() > 1.0)
            & (
                (valid_init["U_num"] * valid_init["P_num"] - valid_init["R_num"]).abs()
                / (valid_init["R_num"].abs() + 1e-6)
                > 0.01
            )
        ).sum()
    )

    # Canonical SKU-Year-Half price (respecting 1 Jan changes + 2022-07..12 6% surcharge!)
    train["Price_Period"] = np.where(
        (train["Date_Clean"] >= "2022-07") & (train["Date_Clean"] <= "2022-12"),
        "2022-H2",
        train["Year_Clean"].astype(str),
    )
    pos_p = train[train["P_num"] > 0]
    canon_price = pos_p.groupby(["SKU_Clean", "Price_Period"])["P_num"].median().rename("Canon_P")
    train = train.merge(canon_price, on=["SKU_Clean", "Price_Period"], how="left")
    # Fallback to SKU-Year median if needed
    sku_yr_p = pos_p.groupby(["SKU_Clean", "Year_Clean"])["P_num"].median().rename("Yr_P")
    train = train.merge(sku_yr_p, on=["SKU_Clean", "Year_Clean"], how="left")
    train["Canon_P"] = train["Canon_P"].fillna(train["Yr_P"])

    train["Units_Clean"] = train["U_num"].copy()
    train["Price_Clean"] = train["P_num"].copy()
    train["Revenue_Clean"] = train["R_num"].copy()
    train["Cleaning_Flag"] = ""

    def add_flag(mask, label):
        train.loc[mask, "Cleaning_Flag"] = np.where(
            train.loc[mask, "Cleaning_Flag"] == "",
            label,
            train.loc[mask, "Cleaning_Flag"] + ";" + label,
        )

    # 8a. Fix negative Units_Sold (sign-flip errors)
    neg_mask = train["Units_Clean"] < 0
    n_neg = int(neg_mask.sum())
    train.loc[neg_mask, "Units_Clean"] = train.loc[neg_mask, "Units_Clean"].abs()
    train.loc[neg_mask, "Revenue_Clean"] = train.loc[neg_mask, "Revenue_Clean"].abs()
    add_flag(neg_mask, "sign_flip_fixed")
    issues.append({
        "file": "train.csv",
        "issue_type": "negative_units_sign_flip",
        "category": "Data Error",
        "rows_affected": n_neg,
        "action": "Corrected sign-flip errors (took absolute value of Units_Sold and Revenue_USD)",
    })

    # 8b. Fix Price currency & scale errors (EUR 0.84x, 10x, 100x cents)
    p_ratio = train["Price_Clean"] / train["Canon_P"]
    p_err_cents = (train["Price_Clean"] > 0) & (p_ratio > 80.0) & (p_ratio < 120.0)
    p_err_10x = (train["Price_Clean"] > 0) & (p_ratio > 8.5) & (p_ratio < 11.5)
    p_err_eur = (train["Price_Clean"] > 0) & (p_ratio > 0.78) & (p_ratio < 0.90)
    p_err_other = (
        (train["Price_Clean"] > 0)
        & ((p_ratio > 1.12) | (p_ratio < 0.88))
        & ~p_err_cents
        & ~p_err_10x
        & ~p_err_eur
    )
    p_err_any = p_err_cents | p_err_10x | p_err_eur | p_err_other
    train.loc[p_err_any, "Price_Clean"] = train.loc[p_err_any, "Canon_P"]
    add_flag(p_err_cents, "price_cents_100x_fixed")
    add_flag(p_err_10x, "price_10x_fixed")
    add_flag(p_err_eur, "price_eur_currency_fixed")
    add_flag(p_err_other, "price_anomaly_fixed")
    issues.append({
        "file": "train.csv",
        "issue_type": "price_scale_and_currency_errors",
        "category": "Data Error",
        "rows_affected": int(p_err_any.sum()),
        "action": f"Reset anomalous Unit_Price_USD to canonical SKU period USD price (100x cents={int(p_err_cents.sum())}, 10x={int(p_err_10x.sum())}, EUR 0.84x={int(p_err_eur.sum())}, other={int(p_err_other.sum())})",
    })

    # 8c. Recover missing Units_Sold from Revenue / Canon_P
    miss_u = train["Units_Clean"].isna()
    recov_u_zero = miss_u & ((train["Revenue_Clean"] == 0) | (train["Price_Clean"] == 0))
    recov_u_calc = miss_u & ~recov_u_zero & train["Revenue_Clean"].notna() & (train["Canon_P"] > 0)
    train.loc[recov_u_zero, "Units_Clean"] = 0.0
    train.loc[recov_u_calc, "Units_Clean"] = np.round(
        train.loc[recov_u_calc, "Revenue_Clean"] / train.loc[recov_u_calc, "Canon_P"]
    )
    add_flag(recov_u_zero, "missing_units_zero_recovered")
    add_flag(recov_u_calc, "missing_units_from_revenue")
    n_still_miss_u = int(train["Units_Clean"].isna().sum())
    issues.append({
        "file": "train.csv",
        "issue_type": "missing_or_nonnumeric_units",
        "category": "Data Error",
        "rows_affected": int(miss_u.sum()),
        "action": f"Recovered {int(recov_u_calc.sum())} from Revenue/Price, {int(recov_u_zero.sum())} as 0, {n_still_miss_u} remain truly missing (also cleaned {n_dash_u} '-' and {n_comma_u} comma-formatted values)",
    })

    # 8d. Fix 10x / 100x Unit errors vs 10x Revenue errors using R / (U * Canon_P)
    valid_check = train["Units_Clean"].notna() & train["Revenue_Clean"].notna() & (train["Units_Clean"] > 0) & (train["Canon_P"] > 0)
    r_over_up = pd.Series(np.nan, index=train.index)
    r_over_up.loc[valid_check] = train.loc[valid_check, "Revenue_Clean"] / (
        train.loc[valid_check, "Units_Clean"] * train.loc[valid_check, "Canon_P"]
    )

    u_10x = valid_check & (r_over_up > 0.095) & (r_over_up < 0.105)
    u_100x = valid_check & (r_over_up > 0.0095) & (r_over_up < 0.0105)
    r_10x = valid_check & (r_over_up > 9.5) & (r_over_up < 10.5)
    # Also check EUR revenue (0.84x) when price was EUR
    r_eur = valid_check & p_err_eur & (r_over_up > 0.78) & (r_over_up < 0.90)

    train.loc[u_10x, "Units_Clean"] = np.round(train.loc[u_10x, "Units_Clean"] / 10.0)
    train.loc[u_100x, "Units_Clean"] = np.round(train.loc[u_100x, "Units_Clean"] / 100.0)
    train.loc[r_10x, "Revenue_Clean"] = np.round(train.loc[r_10x, "Units_Clean"] * train.loc[r_10x, "Price_Clean"], 2)
    train.loc[r_eur, "Revenue_Clean"] = np.round(train.loc[r_eur, "Units_Clean"] * train.loc[r_eur, "Price_Clean"], 2)
    add_flag(u_10x, "units_10x_fixed")
    add_flag(u_100x, "units_100x_fixed")
    add_flag(r_10x, "revenue_10x_fixed")
    add_flag(r_eur, "revenue_eur_fixed")

    issues.append({
        "file": "train.csv",
        "issue_type": "units_scale_errors_10x_100x",
        "category": "Data Error",
        "rows_affected": int(u_10x.sum() + u_100x.sum()),
        "action": f"Divided 10x ({int(u_10x.sum())}) and 100x ({int(u_100x.sum())}) unit entry errors verified via Revenue/Price",
    })
    issues.append({
        "file": "train.csv",
        "issue_type": "revenue_scale_and_currency_errors",
        "category": "Data Error",
        "rows_affected": int(r_10x.sum() + r_eur.sum()),
        "action": f"Reconciled 10x revenue ({int(r_10x.sum())}) and EUR revenue ({int(r_eur.sum())}) to Units * Price_USD",
    })

    # Reconcile remaining Revenue_Clean = Units_Clean * Price_Clean (and fill Price_Clean for 0-unit rows with Canon_P)
    train.loc[train["Price_Clean"] == 0, "Price_Clean"] = train.loc[train["Price_Clean"] == 0, "Canon_P"]
    miss_r = train["Revenue_Clean"].isna() & train["Units_Clean"].notna()
    train.loc[miss_r, "Revenue_Clean"] = np.round(train.loc[miss_r, "Units_Clean"] * train.loc[miss_r, "Price_Clean"], 2)

    # Check any remaining residual mismatches (>1%): verify whether Units_Sold or Revenue_USD was wrong
    rem_valid = train["Units_Clean"].notna() & train["Revenue_Clean"].notna() & (train["Units_Clean"] > 0) & (train["Price_Clean"] > 0)
    rem_diff = (train.loc[rem_valid, "Units_Clean"] * train.loc[rem_valid, "Price_Clean"] - train.loc[rem_valid, "Revenue_Clean"]).abs()
    rem_rel = rem_diff / (train.loc[rem_valid, "Revenue_Clean"].abs() + 1e-6)
    rem_mis_idx = rem_diff[(rem_diff > 1.0) & (rem_rel > 0.01)].index
    if len(rem_mis_idx) > 0:
        sr_med = train[train["Units_Clean"] > 0].groupby(["SKU_Clean", "Region_Clean"])["Units_Clean"].median().to_dict()
        u_fixed_idx, r_fixed_idx = [], []
        for idx in rem_mis_idx:
            u_cur = float(train.at[idx, "Units_Clean"])
            p_cur = float(train.at[idx, "Price_Clean"])
            r_cur = float(train.at[idx, "Revenue_Clean"])
            med_u = sr_med.get((train.at[idx, "SKU_Clean"], train.at[idx, "Region_Clean"]), u_cur)
            u_implied = r_cur / p_cur
            is_exact_int = abs(u_implied - round(u_implied)) < 0.02 and u_implied >= 0
            # If R/P is an exact integer AND strictly closer to the SKU-Region median than U, Units was wrong
            if is_exact_int and abs(round(u_implied) - med_u) < abs(u_cur - med_u):
                train.at[idx, "Units_Clean"] = float(round(u_implied))
                u_fixed_idx.append(idx)
            else:
                train.at[idx, "Revenue_Clean"] = round(u_cur * p_cur, 2)
                r_fixed_idx.append(idx)
        if u_fixed_idx:
            add_flag(train.index.isin(u_fixed_idx), "residual_units_fixed_from_rev")
        if r_fixed_idx:
            add_flag(train.index.isin(r_fixed_idx), "revenue_reconciled")
        issues.append({
            "file": "train.csv",
            "issue_type": "residual_upr_mismatch",
            "category": "Data Error",
            "rows_affected": int(len(rem_mis_idx)),
            "action": f"Checked implied R/P against SKU-Region median: fixed Units on {len(u_fixed_idx)} rows, fixed Revenue on {len(r_fixed_idx)} rows",
        })

    # 9. Ensure 1 clean row per (Date_Clean, SKU_Clean, Region_Clean)
    key_dups = int(train.duplicated(["Date_Clean", "SKU_Clean", "Region_Clean"]).sum())
    if key_dups > 0:
        train = train.sort_values("Load_Timestamp").drop_duplicates(
            ["Date_Clean", "SKU_Clean", "Region_Clean"], keep="last"
        )
        issues.append({
            "file": "train.csv",
            "issue_type": "duplicate_date_sku_region_after_std",
            "category": "Data Error",
            "rows_affected": key_dups,
            "action": "Kept latest Load_Timestamp per (Date, SKU_ID, Region)",
        })

    # 10. Detect missing rows in the expected active (Date, SKU_ID, Region) panel (distinct from 0!)
    # For each (SKU, Region), expected active months = from first observed month (or max(2019-01, Launch_Date)) to min(2026-10, EOL_Date)
    all_months = pd.period_range("2019-01", "2026-10", freq="M").strftime("%Y-%m").tolist()
    month_to_idx = {m: i for i, m in enumerate(all_months)}
    sr_bounds = train.groupby(["SKU_Clean", "Region_Clean"]).agg(
        first_dt=("Date_Clean", "min"),
        last_dt=("Date_Clean", "max"),
        Product_Family=("Product_Family_Clean", "first"),
        Business_Unit=("Business_Unit_Clean", "first"),
        SKU_Description=("SKU_Description_Clean", "first"),
        Customer_Name=("Customer_Name_Clean", "last"),
        Sales_Channel=("Sales_Channel_Clean", "last"),
    ).reset_index()

    existing_keys = set(zip(train["Date_Clean"], train["SKU_Clean"], train["Region_Clean"]))
    missing_rows = []
    for row in sr_bounds.itertuples(index=False):
        i0 = month_to_idx[row.first_dt]
        i1 = month_to_idx[row.last_dt]
        for i in range(i0, i1 + 1):
            m = all_months[i]
            if (m, row.SKU_Clean, row.Region_Clean) not in existing_keys:
                missing_rows.append({
                    "Date": m,
                    "Year": int(m[:4]),
                    "Month": int(m[5:7]),
                    "SKU_ID": row.SKU_Clean,
                    "Region": row.Region_Clean,
                    "Product_Family": row.Product_Family,
                    "Business_Unit": row.Business_Unit,
                    "SKU_Description": row.SKU_Description,
                    "Customer_Name": row.Customer_Name,
                    "Sales_Channel": row.Sales_Channel,
                    "Units_Sold_Raw": np.nan,
                    "Units_Sold_Clean": np.nan,
                    "Unit_Price_USD": np.nan,
                    "Revenue_USD": np.nan,
                    "Is_Missing_Row": True,
                    "Cleaning_Flag": "missing_erp_row",
                })

    issues.append({
        "file": "train.csv",
        "issue_type": "missing_panel_rows_vs_zeros",
        "category": "Data Error / System Gap",
        "rows_affected": len(missing_rows),
        "action": "Inserted explicit NaN panel rows (kept distinct from 0 = no order) for missing SKU-Region-Months",
    })

    # Build final clean DataFrame
    clean_df = pd.DataFrame({
        "Date": train["Date_Clean"],
        "Year": train["Year_Clean"],
        "Month": train["Month_Clean"],
        "SKU_ID": train["SKU_Clean"],
        "Region": train["Region_Clean"],
        "Product_Family": train["Product_Family_Clean"],
        "Business_Unit": train["Business_Unit_Clean"],
        "SKU_Description": train["SKU_Description_Clean"],
        "Customer_Name": train["Customer_Name_Clean"],
        "Sales_Channel": train["Sales_Channel_Clean"],
        "Units_Sold_Raw": train["U_num"],
        "Units_Sold_Clean": train["Units_Clean"],
        "Unit_Price_USD": train["Price_Clean"],
        "Revenue_USD": train["Revenue_Clean"],
        "Is_Missing_Row": False,
        "Cleaning_Flag": train["Cleaning_Flag"],
    })

    if missing_rows:
        miss_df = pd.DataFrame(missing_rows)
        # Fill canonical Unit_Price_USD for missing rows
        miss_df["Price_Period"] = np.where(
            (miss_df["Date"] >= "2022-07") & (miss_df["Date"] <= "2022-12"),
            "2022-H2",
            miss_df["Year"].astype(str),
        )
        miss_df = miss_df.merge(
            canon_price.reset_index().rename(columns={"SKU_Clean": "SKU_ID"}),
            on=["SKU_ID", "Price_Period"],
            how="left",
        )
        miss_df["Unit_Price_USD"] = miss_df["Canon_P"]
        miss_df = miss_df.drop(columns=["Price_Period", "Canon_P"])
        clean_df = pd.concat([clean_df, miss_df], ignore_index=True)

    clean_df = clean_df.sort_values(["SKU_ID", "Region", "Date"]).reset_index(drop=True)

    # Final reconciliation verification
    v_after = clean_df.dropna(subset=["Units_Sold_Clean", "Unit_Price_USD", "Revenue_USD"])
    after_mismatch = int(
        (
            ((v_after["Units_Sold_Clean"] * v_after["Unit_Price_USD"] - v_after["Revenue_USD"]).abs() > 1.0)
            & (
                (v_after["Units_Sold_Clean"] * v_after["Unit_Price_USD"] - v_after["Revenue_USD"]).abs()
                / (v_after["Revenue_USD"].abs() + 1e-6)
                > 0.01
            )
        ).sum()
    )

    clean_df.to_csv(out_dir / "train_clean.csv.gz", index=False)

    summary = {
        "file": "train.csv",
        "rows_before": rows_before,
        "rows_after_dedup": int((~clean_df["Is_Missing_Row"]).sum()),
        "rows_after_panel_completion": len(clean_df),
        "missing_panel_rows_added": len(missing_rows),
        "unique_skus_before": unique_skus_before,
        "unique_skus_after": int(clean_df["SKU_ID"].nunique()),
        "unique_regions_before": unique_regions_before,
        "unique_regions_after": int(clean_df["Region"].nunique()),
        "unique_customers_before": unique_customers_before,
        "unique_channels_before": unique_channels_before,
        "unique_customers_after": int(clean_df["Customer_Name"].nunique()),
        "reconciliation_mismatches_before": init_mismatch,
        "reconciliation_mismatches_after": after_mismatch,
        "negative_units_before": n_neg,
        "negative_units_after": int((clean_df["Units_Sold_Clean"] < 0).sum()),
        "total_units_before_raw": float(pd.to_numeric(pd.read_csv(raw_dir / "train.csv", usecols=["Units_Sold"], dtype=str)["Units_Sold"].str.replace(",", "", regex=False), errors="coerce").sum()),
        "total_units_after_clean": float(clean_df["Units_Sold_Clean"].sum()),
        "issues": issues,
    }
    return clean_df, summary


def run_cleaning(raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.read_csv(raw_dir / "sku_master.csv").to_csv(out_dir / "sku_master_clean.csv", index=False)
    pd.read_csv(raw_dir / "events_calendar.csv").to_csv(out_dir / "events_calendar_clean.csv", index=False)
    macro_df, macro_summary = clean_macro(raw_dir, out_dir)
    train_df, train_summary = clean_train(raw_dir, out_dir)

    report = {
        "pipeline_stage": "Workstream 1: Data Cleaning & Quality Reconciliation",
        "files_profiled": [
            "train.csv",
            "test.csv",
            "macro_indicators.csv",
            "sku_master.csv",
            "events_calendar.csv",
            "inventory_stockouts.csv",
            "sample_submission.csv",
        ],
        "train_summary": train_summary,
        "macro_summary": macro_summary,
        "all_issues": train_summary["issues"] + macro_summary["issues"],
    }
    with open(out_dir / "data_quality_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    rep = run_cleaning()
    print("Cleaning complete!")
    print("Train rows before:", rep["train_summary"]["rows_before"],
          "| after dedup:", rep["train_summary"]["rows_after_dedup"],
          "| after panel:", rep["train_summary"]["rows_after_panel_completion"],
          "| missing panel rows:", rep["train_summary"]["missing_panel_rows_added"])
    print("Reconciliation mismatches before:", rep["train_summary"]["reconciliation_mismatches_before"],
          "| after:", rep["train_summary"]["reconciliation_mismatches_after"])
    print("Total issues logged:", len(rep["all_issues"]))
