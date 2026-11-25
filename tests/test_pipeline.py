import json
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "outputs"
RAW_DIR = PROJECT_DIR / "data" / "raw" / "student_data"
sys.path.insert(0, str(PROJECT_DIR / "dashboard"))

from app import DataStore


class TestCelerisPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_raw = pd.read_csv(RAW_DIR / "test.csv")
        cls.sub = pd.read_csv(OUT_DIR / "submission.csv")
        cls.train_clean = pd.read_csv(OUT_DIR / "train_clean.csv.gz", low_memory=False)
        cls.train_enr = pd.read_csv(OUT_DIR / "train_enriched.csv.gz", low_memory=False)
        cls.fc_full = pd.read_csv(OUT_DIR / "forecast_full.csv.gz", low_memory=False)
        cls.macro = pd.read_csv(OUT_DIR / "macro_clean.csv")
        with open(OUT_DIR / "data_quality_report.json") as f:
            cls.dq = json.load(f)
        with open(OUT_DIR / "anomaly_summary.json") as f:
            cls.anom = json.load(f)
        with open(OUT_DIR / "macro_lead_lag.json") as f:
            cls.drivers = json.load(f)
        with open(OUT_DIR / "backtest_metrics.json") as f:
            cls.bt = json.load(f)

    def test_01_cleaning_invariants_and_dynamic_counts(self):
        self.assertEqual(self.train_clean.duplicated(["Date", "SKU_ID", "Region"]).sum(), 0)
        self.assertEqual((self.train_clean["Units_Sold_Clean"] < 0).sum(), 0)
        self.assertEqual(self.dq["train_summary"]["reconciliation_mismatches_after"], 0)
        self.assertIn("unique_channels_before", self.dq["train_summary"])
        miss = self.train_clean[self.train_clean["Is_Missing_Row"]]
        self.assertGreater(len(miss), 0)
        self.assertTrue(miss["Units_Sold_Clean"].isna().all())

    def test_02_macro_clean_and_covid_excluded_drivers(self):
        self.assertEqual(len(self.macro), 113)
        self.assertEqual(self.macro.isna().sum().sum(), 0)
        self.assertIn("2020-02", self.drivers["covid_window_excluded"])
        self.assertIn("ev_share_vs_adoption_detrended_corr", self.drivers)
        self.assertLess(abs(self.drivers["ev_share_vs_adoption_detrended_corr"]), 0.85)

    def test_03_stockout_backorder_catchup_and_detector_recall(self):
        ss = self.anom["stockout_summary"]
        self.assertGreater(ss["backorder_catchup_units_subtracted"], 10000)
        self.assertGreater(ss["detector_validation_recall_overall_pct"], 40.0)
        self.assertLess(ss["general_unrecorded_detection_rate_pct"], 1.0)

    def test_04_submission_schema_and_orion_full_cut(self):
        self.assertEqual(list(self.sub.columns), ["Date", "SKU_ID", "Region", "Units_Sold_Forecast"])
        self.assertEqual(len(self.sub), len(self.test_raw))
        self.assertEqual(self.sub.isna().sum().sum(), 0)
        self.assertTrue((self.sub["Units_Sold_Forecast"] >= 0).all())
        self.assertTrue(pd.api.types.is_integer_dtype(self.sub["Units_Sold_Forecast"]))
        self.assertTrue((self.sub["Date"].values == self.test_raw["Date"].values).all())
        self.assertTrue((self.sub["SKU_ID"].values == self.test_raw["SKU_ID"].values).all())
        self.assertTrue((self.sub["Region"].values == self.test_raw["Region"].values).all())
        # Verify Orion EV-062 note reflects full post-blend cut
        orion = self.fc_full[
            (self.fc_full["Customer_Name"] == "Orion Driveline")
            & (self.fc_full["Region"] == "North America")
            & (self.fc_full["Date"].isin(["2027-07", "2027-08"]))
        ]
        self.assertGreater(len(orion), 0)
        self.assertTrue(orion["Lifecycle_Note"].str.contains("post-blend").all())

    def test_05_lifecycle_and_4origin_reconciled_backtest(self):
        for sku_id, eol in [("PMP-003", "2027-08"), ("PTC-005", "2027-05"), ("THM-001", "2027-12")]:
            post = self.sub[(self.sub["SKU_ID"] == sku_id) & (self.sub["Date"] > eol)]
            self.assertEqual(post["Units_Sold_Forecast"].sum(), 0)
        for sku_id in ["PMP-020", "PTC-029", "THM-012"]:
            self.assertGreater(self.sub[self.sub["SKU_ID"] == sku_id]["Units_Sold_Forecast"].sum(), 0)

        # 4 rolling origins & reconciled model beats baselines
        self.assertEqual(len(self.bt["origins"]), 4)
        mc = self.bt["model_comparison"]
        self.assertLess(
            mc["Reconciled & Rounded Ensemble (Final Submission)"]["WAPE_Pct"],
            mc["Seasonal Naive (Baseline 1)"]["WAPE_Pct"],
        )

    def test_06_oos_interval_calibration_sku_and_aggregate(self):
        ic = self.bt["interval_calibration"]
        sku_cov = ic["sku_level"]["oos_coverage_pct"]
        self.assertGreaterEqual(sku_cov, 75.0)
        self.assertLessEqual(sku_cov, 88.0)
        # Aggregate global P90 must be much tighter than naive SKU-summed P90
        hl = self.bt["forecast_headline"]
        self.assertLess(hl["total_p90_units_global_calibrated"], hl["sku_sum_p90_units_for_sku_level_buffer"])
        self.assertTrue((self.fc_full["P10"] <= self.fc_full["P50"]).all())
        self.assertTrue((self.fc_full["P50"] <= self.fc_full["P90"]).all())

    def test_07_dashboard_calibrated_aggregates_and_model_whatif(self):
        store = DataStore(OUT_DIR, RAW_DIR)
        ov = store.get_overview({})
        self.assertEqual(ov["kpis"]["interval_level"], "global")
        # Global P90 should be ~15-25% above P50, NOT +52%!
        self.assertLess(ov["kpis"]["fc_p90_units"] / ov["kpis"]["fc_p50_units"], 1.30)
        wi = store.get_what_if({"pmi_delta": ["-3"]})
        self.assertIn("elasticities", wi)
        self.assertLess(wi["unit_diff"], 0)
        self.assertGreater(store.get_procurement({})["total_p90"], store.get_procurement({})["total_p50"])
        self.assertEqual(len(store.get_lifecycle({})["active_transitions"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
