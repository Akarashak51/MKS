# Celeris Automobili Industrial — Demand Forecasting & Planning Control Tower (v7 Final)

End-to-end demand forecasting pipeline and interactive **12-view Planning Control Tower** covering all 5 workstreams across Core, Advanced, and Expert tiers.

## Quick Start (Local Setup)

### Requirements
- Python 3.10+ with `pandas` and `numpy` (`pip install pandas numpy`) — **no `pyarrow` required**.

### 1. Launch the Planning Control Tower Dashboard (Works Immediately Out of the Box)
```bash
cd celeris_control_tower
python3 dashboard/app.py --port 8050
```
Open `http://127.0.0.1:8050` in your browser. All 12 views share cascading global filters (`Product Family → SKU`, `Region`, `Sales Channel`, `Customer`), read live from `outputs/`, and support **Dark Black** and **Dark White** themes.

### 2. Reproduce the Full Pipeline & Run Automated Tests (Requires Raw Competition CSVs)
Place the 7 raw CSV files inside `celeris_control_tower/data/raw/student_data/` and run:
```bash
python3 run_pipeline.py
python3 -m unittest tests/test_pipeline.py -v
```

## Key Results Summary (100% Synced to v7 Pipeline Outputs)

### Workstream 1 — Data Cleaning & Financial Reconciliation
- **ERP Sales (`train.csv`)**: `248,248` raw → `244,557` deduplicated → `245,522` panel-complete rows (`965` explicit `NaN` missing ERP rows kept distinct from `0` = no order).
- **Dynamic Taxonomy Audit**: `2,659` raw SKUs → `598`, `30` raw regions → `6`, `17` raw channels → `4`, `220` raw customers → `12`.
- **Financial Reconciliation**: `Units × Price ≠ Revenue` mismatches reduced **`1,591` → `0`**; `491` sign-flipped negative orders fixed; `macro_indicators.csv` cleaned to `113` unique months (`0` nulls, `2023-05` PMI `4.79 → 47.9`, 2019 Auto YoY scaled).

### Workstream 2 — Anomalies, Backorder Catch-Up & Recall-Validated Stockouts
- **Anomaly Register**: `10,952` entries separating `7,437` ERP errors from `3,515` real events and `16` `BSS-018` MDM contradictions.
- **Backorder Catch-Up De-Duplication**: Subtracted **`13,282` catch-up units** across `1,048` post-backorder months.
- **Tightened Stockout Detector**: `5,830` recorded + `1,640` detected unrecorded (`902` EV-019 semiconductor + `738` general at `>=50%` drop, `>=2σ`, healthy-neighbor check). Validation recall is **`47.4%`** with a conservative **`0.31%`** non-shortage detection rate; net unconstrained uplift **`+124,187` units**.

### Workstream 3 — COVID-Excluded Drivers & EV Shift
- **COVID-Excluded Macro Lead/Lag (`2020-02` to `2021-06` masked)**: Discovers genuine **2–3 month** family-specific OEM booking leads (`Auto_Production_YoY_Pct` & `New_Vehicle_Orders_Index`) fed directly into the forecasting model.
- **EV Transition**: EV-dedicated share rose **`0.0%` (2019) → `9.81%` (2026)** while ICE-only fell **`24.08%` → `13.05%`** (`r = 0.976` level; **`r = 0.414`** detrended MoM diff).

### Workstream 4 — 4-Origin 19M Backtest, Multi-Level Conformal Calibration & Submission
- **SKU-Region 4-Origin Backtest**: Reconciled & Rounded Ensemble **`35.19%` WAPE** (`-4.86%` Bias; `-4.00%` vs. shipped actuals), beating Seasonal Naive (`37.49%`) by **1.8–2.8 points (2.30 points pooled)** and MA-6 (`51.29%`) by **16.10 points**.
- **Aggregate Backtest Accuracy**: **Global Monthly Total = `7.52%` WAPE** (vs. `8.42%` SN) | **Family-Region Monthly = `11.29%` WAPE** (vs. `11.93%` SN).
- **Reconciliation**: 90/10 proportional pull toward prior-year same-month Family-Region volume (clipped `±8%`) + largest-remainder integer rounding.
- **Per-Origin Bias Diagnostic** (pooled `-4.86%` reflects 2023–2024 growth-phase origins; latest origin sits at `+1.54%`, with family-level latest-origin bias within `±8%`):

| Origin | 19-Month Window | Reconciled WAPE | Reconciled Bias | Seasonal Naive WAPE | Seasonal Naive Bias |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `2023-09` | `2023-10..2025-04` | **37.03%** | **-9.02%** | 39.46% | -8.96% |
| `2024-03` | `2024-04..2025-10` | **35.22%** | **-10.64%** | 37.02% | -8.83% |
| `2024-09` | `2024-10..2026-04` | **34.18%** | **-1.19%** | 36.39% | -3.90% |
| `2025-03` | `2025-04..2026-10` | **34.31%** | **+1.54%** | 37.11% | +4.08% |

- **Out-of-Sample Multi-Level Interval Calibration (Calibrated on Origins 1–3, Validated OOS on Origin 4)**:
  We provide **both** a tight **P10–P90 Planning Band** (centered split-conformal, targeting ≥80% OOS) and a **P01–P99 Tail-Risk Stress Envelope** (targeting 99% OOS, achieving **99.3%–100.0%** across all 5 hierarchy levels):

| Hierarchy Level | P10–P90 OOS Cov | P10–P90 Width | P01–P99 Stress Cov | OOS Points |
| :--- | ---: | ---: | ---: | ---: |
| **SKU-Region Monthly** | **82.2%** | 124.5% | **99.3%** | 57,830 |
| **Global Monthly Total** | **100.0%** | 26.2% | **100.0%** | 19 |
| **Product Family Monthly** | **94.0%** | 34.3% | **100.0%** | 133 |
| **Region Monthly** | **91.2%** | 27.7% | **100.0%** | 114 |
| **Family-Region Monthly** | **91.2%** | 42.0% | **100.0%** | 798 |

- **19-Month Forecast (`outputs/submission.csv`)**: `58,219` non-negative integer rows, `0` blanks, **`2,632,847` P50 units (`$2,366,708,478`)**, calibrated **Portfolio P90 = `2,972,749` units (`$2,672,250,542`)** (`+12.9%` aggregate plant buffer), while individual SKU-region P90 buffers sum to `4,415,502` units. Enforces strict `0` post-EOL (`PMP-003`, `PTC-005`, `THM-001`), calibrated `Replaces_SKU` launches (`PMP-020`, `PTC-029`, `THM-012`), and full 55% post-blend `EV-062` Orion shutdown.
- **Model-Derived What-If Elasticities**: Extracted from fitted Ridge weights (`PMI +1.03%/pt`, `Auto YoY +0.27%/%`, `Price -1.28%/%`).
