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
        cls.sku_master = pd.read_csv(RAW_DIR / "sku_master.csv")
        with open(OUT_DIR / "data_quality_report.json") as f:
            cls.dq = json.load(f)
        with open(OUT_DIR / "backtest_metrics.json") as f:
            cls.bt = json.load(f)

    def test_01_cleaning_invariants(self):
        # One clean row per (Date, SKU_ID, Region)
        self.assertEqual(self.train_clean.duplicated(["Date", "SKU_ID", "Region"]).sum(), 0)
        # Zero negative units
        self.assertEqual((self.train_clean["Units_Sold_Clean"] < 0).sum(), 0)
        # Zero reconciliation mismatches
        self.assertEqual(self.dq["train_summary"]["reconciliation_mismatches_after"], 0)
        # Canonical counts
        self.assertEqual(self.train_clean["Region"].nunique(), 6)
        self.assertEqual(self.train_clean["Sales_Channel"].nunique(), 4)
        self.assertEqual(self.train_clean["Customer_Name"].nunique(), 12)
        # Missing panel rows kept distinct from 0
        miss = self.train_clean[self.train_clean["Is_Missing_Row"]]
        self.assertGreater(len(miss), 0)
        self.assertTrue(miss["Units_Sold_Clean"].isna().all())

    def test_02_macro_clean(self):
        self.assertEqual(len(self.macro), 113)
        self.assertEqual(self.macro["Date"].nunique(), 113)
        self.assertEqual(self.macro.isna().sum().sum(), 0)
        # 2023-05 PMI fixed from 4.79 -> 47.9
        pmi_2305 = self.macro.loc[self.macro["Date"] == "2023-05", "ISM_Manufacturing_PMI"].iloc[0]
        self.assertGreater(pmi_2305, 40.0)
        # 2019 Auto_Production_YoY_Pct fixed from decimal to percent
        auto_19 = self.macro.loc[self.macro["Year"] == 2019, "Auto_Production_YoY_Pct"].abs().mean()
        self.assertGreater(auto_19, 1.0)

    def test_03_stockout_unconstraining(self):
        st = self.train_enr[self.train_enr["Stockout_Status"] != "none"]
        self.assertGreater((st["Stockout_Status"] == "recorded").sum(), 5000)
        self.assertGreater((st["Stockout_Status"] == "detected_unrecorded").sum(), 500)
        # Unconstrained >= Shipped on detected unrecorded stockouts
        det = st[st["Stockout_Status"] == "detected_unrecorded"]
        self.assertTrue((det["Unconstrained_Demand"] >= det["Units_Sold_Clean"]).all())

    def test_04_submission_schema_and_constraints(self):
        self.assertEqual(list(self.sub.columns), ["Date", "SKU_ID", "Region", "Units_Sold_Forecast"])
        self.assertEqual(len(self.sub), len(self.test_raw))
        self.assertEqual(self.sub.isna().sum().sum(), 0)
        self.assertTrue((self.sub["Units_Sold_Forecast"] >= 0).all())
        self.assertTrue(pd.api.types.is_integer_dtype(self.sub["Units_Sold_Forecast"]))
        # Exact key alignment with test.csv
        self.assertTrue((self.sub["Date"].values == self.test_raw["Date"].values).all())
        self.assertTrue((self.sub["SKU_ID"].values == self.test_raw["SKU_ID"].values).all())
        self.assertTrue((self.sub["Region"].values == self.test_raw["Region"].values).all())

    def test_05_lifecycle_and_intervals_and_hierarchy(self):
        # Post-EOL SKUs must be strictly 0 after EOL_Date
        for sku_id, eol in [("PMP-003", "2027-08"), ("PTC-005", "2027-05"), ("THM-001", "2027-12")]:
            post = self.sub[(self.sub["SKU_ID"] == sku_id) & (self.sub["Date"] > eol)]
            self.assertGreater(len(post), 0)
            self.assertEqual(post["Units_Sold_Forecast"].sum(), 0, f"{sku_id} has non-zero post-EOL sales")

        # Planned launches must have positive forecast from Launch_Date
        for sku_id in ["PMP-020", "PTC-029", "THM-012"]:
            pl = self.sub[self.sub["SKU_ID"] == sku_id]
            self.assertGreater(pl["Units_Sold_Forecast"].sum(), 0, f"Planned launch {sku_id} has 0 forecast")

        # P10 <= P50 <= P90
        self.assertTrue((self.fc_full["P10"] <= self.fc_full["P50"]).all())
        self.assertTrue((self.fc_full["P50"] <= self.fc_full["P90"]).all())

        # Exact hierarchical additivity
        self.assertEqual(self.bt["reconciliation"]["max_hierarchy_discrepancy"], 0)

    def test_06_backtest_beats_baselines(self):
        mc = self.bt["model_comparison"]
        ens_wape = mc["Final Ensemble (SBC + Lifecycle)"]["WAPE_Pct"]
        sn_wape = mc["Seasonal Naive (Baseline 1)"]["WAPE_Pct"]
        ma_wape = mc["Moving Average 6M (Baseline 2)"]["WAPE_Pct"]
        self.assertLess(ens_wape, sn_wape)
        self.assertLess(ens_wape, ma_wape)

    def test_07_dashboard_datastore_endpoints(self):
        store = DataStore(OUT_DIR, RAW_DIR)
        empty = {}
        self.assertGreater(len(store.get_overview(empty)["timeline"]), 100)
        self.assertGreater(len(store.get_demand_explorer(empty)["drilldown"]), 0)
        self.assertGreater(store.get_anomalies(empty)["filtered_count"], 0)
        self.assertLess(store.get_what_if({"pmi_delta": ["-3"]})["unit_diff"], 0)
        self.assertGreater(store.get_procurement(empty)["total_p90"], store.get_procurement(empty)["total_p50"])
        self.assertEqual(len(store.get_lifecycle(empty)["active_transitions"]), 3)
        self.assertGreater(store.get_stockouts(empty)["summary"]["detected_rows"], 0)
        self.assertGreater(store.get_exceptions(empty)["total_exceptions"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
