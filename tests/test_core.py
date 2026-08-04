import numpy as np

from pmsm_twin.core import FOCPI, PMSM, PMSMParams, PredictiveCurrentControl, run_episode


def test_zero_state_without_voltage_or_load_is_equilibrium():
    assert np.allclose(PMSM().derivative(np.zeros(4), np.zeros(2), 0.0), 0.0)


def test_positive_iq_produces_positive_torque():
    assert PMSM().torque(np.array([0.0, 2.0, 0.0, 0.0])) > 0


def test_controllers_respect_voltage_disk():
    params = PMSMParams()
    state = np.array([4, -2, 150, 0])
    for controller in [FOCPI(params), PredictiveCurrentControl(params)]:
        assert (
            np.linalg.norm(controller.command(state, 100, 0.0002)) <= params.phase_voltage_limit + 1e-9
        )


def test_predictive_controller_tracks_and_never_violates_voltage_limit():
    result = run_episode("predictive", duration_s=0.65)
    assert result["voltage_limit_violations"] == 0
    assert result["speed_rmse_percent_rated"] < 20
