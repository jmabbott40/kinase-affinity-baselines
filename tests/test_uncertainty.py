"""Tests for uncertainty calibration and analysis functions."""

import numpy as np
import pandas as pd
import pytest

from kinase_affinity.evaluation.uncertainty import (
    calibration_curve,
    error_uncertainty_correlation,
    miscalibration_area,
    selective_prediction_curve,
)
from kinase_affinity.evaluation.analysis import (
    find_worst_predictions,
    noise_impact_analysis,
    per_target_metrics,
)


# ---- Fixtures ----

@pytest.fixture
def well_calibrated_data():
    """Generate data where y_std matches actual error distribution."""
    rng = np.random.RandomState(42)
    n = 5000
    y_true = rng.normal(7.0, 1.5, n)
    # Heteroscedastic noise: std varies across samples
    y_std = np.abs(rng.normal(0.5, 0.2, n)) + 0.1
    noise = rng.normal(0, 1, n) * y_std  # noise scaled by std
    y_pred = y_true + noise
    return y_true, y_pred, y_std


@pytest.fixture
def overconfident_data():
    """Model that underestimates its uncertainty (intervals too narrow)."""
    rng = np.random.RandomState(42)
    n = 5000
    y_true = rng.normal(7.0, 1.5, n)
    actual_std = 1.0
    reported_std = np.full(n, 0.2)  # Claims much lower uncertainty
    noise = rng.normal(0, actual_std, n)
    y_pred = y_true + noise
    return y_true, y_pred, reported_std


# ---- Calibration curve tests ----

class TestCalibrationCurve:
    def test_perfect_calibration(self, well_calibrated_data):
        y_true, y_pred, y_std = well_calibrated_data
        expected, observed = calibration_curve(y_true, y_pred, y_std, n_bins=10)

        assert len(expected) == 10
        assert len(observed) == 10
        # Well-calibrated: observed should track expected within ~5%
        assert np.max(np.abs(expected - observed)) < 0.10

    def test_overconfident_detection(self, overconfident_data):
        y_true, y_pred, y_std = overconfident_data
        expected, observed = calibration_curve(y_true, y_pred, y_std, n_bins=10)

        # Overconfident: observed coverage should be LESS than expected
        # (intervals are too narrow, so fewer true values fall inside)
        assert np.mean(observed < expected) > 0.5

    def test_output_shape(self, well_calibrated_data):
        y_true, y_pred, y_std = well_calibrated_data
        expected, observed = calibration_curve(y_true, y_pred, y_std, n_bins=15)
        assert expected.shape == (15,)
        assert observed.shape == (15,)

    def test_expected_range(self, well_calibrated_data):
        y_true, y_pred, y_std = well_calibrated_data
        expected, observed = calibration_curve(y_true, y_pred, y_std)
        assert np.all(expected > 0)
        assert np.all(expected < 1)
        assert np.all(observed >= 0)
        assert np.all(observed <= 1)


# ---- Miscalibration area tests ----

class TestMiscalibrationArea:
    def test_perfect_calibration(self):
        x = np.linspace(0.1, 0.9, 10)
        area = miscalibration_area(x, x)  # Perfect: observed == expected
        assert area < 0.01

    def test_worst_calibration(self):
        expected = np.linspace(0.1, 0.9, 10)
        observed = np.zeros(10)  # Worst case
        area = miscalibration_area(expected, observed)
        assert area > 0.1

    def test_well_calibrated_is_small(self, well_calibrated_data):
        y_true, y_pred, y_std = well_calibrated_data
        expected, observed = calibration_curve(y_true, y_pred, y_std)
        area = miscalibration_area(expected, observed)
        assert area < 0.05  # Should be small for well-calibrated


# ---- Error-uncertainty correlation tests ----

class TestErrorUncertaintyCorrelation:
    def test_informative_uncertainty(self):
        """When uncertainty tracks error, correlations should be positive."""
        rng = np.random.RandomState(42)
        n = 1000
        y_true = rng.normal(7, 1, n)
        y_std = np.abs(rng.normal(0.5, 0.3, n)) + 0.1
        noise = rng.normal(0, 1, n) * y_std
        y_pred = y_true + noise

        result = error_uncertainty_correlation(y_true, y_pred, y_std)
        assert result["pearson_r"] > 0.2
        assert result["spearman_rho"] > 0.2

    def test_uninformative_uncertainty(self):
        """Constant uncertainty should give NaN correlations."""
        rng = np.random.RandomState(42)
        n = 100
        y_true = rng.normal(7, 1, n)
        y_pred = y_true + rng.normal(0, 0.5, n)
        y_std = np.full(n, 0.5)  # Constant → no correlation possible

        result = error_uncertainty_correlation(y_true, y_pred, y_std)
        assert np.isnan(result["pearson_r"])

    def test_returns_all_keys(self):
        rng = np.random.RandomState(42)
        n = 100
        y_true = rng.normal(7, 1, n)
        y_pred = y_true + rng.normal(0, 0.5, n)
        y_std = np.abs(rng.normal(0.5, 0.2, n)) + 0.1

        result = error_uncertainty_correlation(y_true, y_pred, y_std)
        assert "pearson_r" in result
        assert "pearson_p" in result
        assert "spearman_rho" in result
        assert "spearman_p" in result


# ---- Selective prediction tests ----

class TestSelectivePrediction:
    def test_rmse_decreases_with_selective_rejection(self):
        """RMSE should decrease when rejecting high-uncertainty predictions."""
        rng = np.random.RandomState(42)
        n = 2000
        y_true = rng.normal(7, 1, n)
        y_std = np.abs(rng.normal(0.5, 0.3, n)) + 0.1
        noise = rng.normal(0, 1, n) * y_std
        y_pred = y_true + noise

        retention, rmse = selective_prediction_curve(y_true, y_pred, y_std)
        # RMSE at 50% retention should be lower than at 100%
        assert rmse[9] < rmse[-1]

    def test_output_shape(self):
        rng = np.random.RandomState(42)
        n = 100
        y_true = rng.normal(7, 1, n)
        y_pred = y_true + rng.normal(0, 0.5, n)
        y_std = np.abs(rng.normal(0.5, 0.2, n)) + 0.1

        retention, rmse = selective_prediction_curve(y_true, y_pred, y_std, n_points=15)
        assert retention.shape == (15,)
        assert rmse.shape == (15,)

    def test_retention_range(self):
        rng = np.random.RandomState(42)
        n = 100
        y_true = rng.normal(7, 1, n)
        y_pred = y_true + rng.normal(0, 0.5, n)
        y_std = np.abs(rng.normal(0.5, 0.2, n)) + 0.1

        retention, rmse = selective_prediction_curve(y_true, y_pred, y_std)
        assert retention[-1] == 1.0
        assert retention[0] > 0
        assert np.all(rmse >= 0)


# ---- Error analysis tests ----

class TestFindWorstPredictions:
    def test_returns_correct_count(self):
        n = 100
        y_true = np.arange(n, dtype=float)
        y_pred = y_true + np.random.RandomState(42).normal(0, 1, n)
        df = pd.DataFrame({"std_smiles": [f"C{i}" for i in range(n)]})

        worst = find_worst_predictions(y_true, y_pred, df, top_n=10)
        assert len(worst) == 10

    def test_sorted_by_error(self):
        n = 50
        y_true = np.zeros(n)
        y_pred = np.arange(n, dtype=float)  # Errors increase with index
        df = pd.DataFrame({"std_smiles": [f"C{i}" for i in range(n)]})

        worst = find_worst_predictions(y_true, y_pred, df, top_n=5)
        assert worst["abs_error"].iloc[0] >= worst["abs_error"].iloc[-1]

    def test_contains_metadata(self):
        n = 20
        y_true = np.ones(n) * 7.0
        y_pred = np.ones(n) * 6.0
        df = pd.DataFrame({
            "std_smiles": [f"C{i}" for i in range(n)],
            "target_chembl_id": [f"T{i}" for i in range(n)],
        })

        worst = find_worst_predictions(y_true, y_pred, df, top_n=5)
        assert "y_true" in worst.columns
        assert "y_pred" in worst.columns
        assert "abs_error" in worst.columns
        assert "signed_error" in worst.columns
        assert "std_smiles" in worst.columns


class TestPerTargetMetrics:
    def test_basic_breakdown(self):
        rng = np.random.RandomState(42)
        # Two targets with different performance
        n = 200
        target_ids = np.array(["T1"] * 100 + ["T2"] * 100)
        y_true = rng.normal(7, 1, n)
        y_pred = y_true.copy()
        # T2 has much worse predictions
        y_pred[100:] += rng.normal(0, 2, 100)

        result = per_target_metrics(y_true, y_pred, target_ids, min_samples=10)
        assert len(result) == 2
        # T1 should have lower RMSE than T2
        t1_rmse = result[result["target_id"] == "T1"]["rmse"].values[0]
        t2_rmse = result[result["target_id"] == "T2"]["rmse"].values[0]
        assert t1_rmse < t2_rmse

    def test_min_samples_filter(self):
        target_ids = np.array(["T1"] * 50 + ["T2"] * 5)
        y_true = np.random.RandomState(42).normal(7, 1, 55)
        y_pred = y_true + 0.1

        result = per_target_metrics(y_true, y_pred, target_ids, min_samples=10)
        assert len(result) == 1  # T2 filtered out
        assert result["target_id"].iloc[0] == "T1"


class TestNoiseImpactAnalysis:
    def test_basic_comparison(self):
        rng = np.random.RandomState(42)
        n = 200
        y_true = rng.normal(7, 1, n)
        y_pred = y_true + rng.normal(0, 0.5, n)
        is_noisy = np.zeros(n, dtype=bool)
        is_noisy[:20] = True
        # Make noisy predictions worse
        y_pred[:20] += rng.normal(0, 2, 20)

        result = noise_impact_analysis(y_true, y_pred, is_noisy)
        assert result["n_clean"] == 180
        assert result["n_noisy"] == 20
        assert "clean" in result
        assert "noisy" in result
        assert result["noisy"]["rmse"] > result["clean"]["rmse"]

    def test_no_noisy_samples(self):
        n = 50
        y_true = np.ones(n) * 7.0
        y_pred = np.ones(n) * 6.5
        is_noisy = np.zeros(n, dtype=bool)

        result = noise_impact_analysis(y_true, y_pred, is_noisy)
        assert result["n_noisy"] == 0
        assert result["n_clean"] == 50
