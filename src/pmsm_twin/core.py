"""Continuous dq-frame PMSM model with voltage-constrained controllers."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class PMSMParams:
    resistance_ohm: float = 0.45
    ld_h: float = 0.004
    lq_h: float = 0.0045
    flux_wb: float = 0.115
    pole_pairs: int = 4
    inertia_kgm2: float = 0.003
    viscous_friction: float = 0.0004
    dc_bus_v: float = 300.0
    current_limit_a: float = 20.0

    def drifted(self, resistance_scale: float = 1.0, flux_scale: float = 1.0) -> PMSMParams:
        return replace(
            self,
            resistance_ohm=self.resistance_ohm * resistance_scale,
            flux_wb=self.flux_wb * flux_scale,
        )

    @property
    def phase_voltage_limit(self) -> float:
        return self.dc_bus_v / np.sqrt(3)


class PMSM:
    """State is [id, iq, mechanical speed, electrical angle]."""

    def __init__(self, params: PMSMParams | None = None):
        self.params = PMSMParams() if params is None else params

    def torque(self, state: Array) -> float:
        i_d, i_q = state[:2]
        p = self.params
        return float(1.5 * p.pole_pairs * (p.flux_wb * i_q + (p.ld_h - p.lq_h) * i_d * i_q))

    def derivative(self, state: Array, voltage: Array, load_torque_nm: float) -> Array:
        i_d, i_q, omega, _ = np.asarray(state, dtype=float)
        v_d, v_q = voltage
        p = self.params
        omega_e = p.pole_pairs * omega
        di_d = (v_d - p.resistance_ohm * i_d + omega_e * p.lq_h * i_q) / p.ld_h
        di_q = (v_q - p.resistance_ohm * i_q - omega_e * (p.ld_h * i_d + p.flux_wb)) / p.lq_h
        domega = (self.torque(state) - load_torque_nm - p.viscous_friction * omega) / p.inertia_kgm2
        return np.array([di_d, di_q, domega, omega_e])

    def step(self, state: Array, voltage: Array, load_torque_nm: float, dt: float) -> Array:
        limit = self.params.phase_voltage_limit
        norm = np.linalg.norm(voltage)
        applied = np.asarray(voltage, dtype=float) * min(1.0, limit / max(norm, 1e-12))
        f = lambda s: self.derivative(s, applied, load_torque_nm)
        k1 = f(state)
        k2 = f(state + dt * k1 / 2)
        k3 = f(state + dt * k2 / 2)
        k4 = f(state + dt * k3)
        nxt = np.asarray(state) + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        nxt[3] %= 2 * np.pi
        return nxt


class SpeedReference:
    def __init__(self, params: PMSMParams):
        self.params = params
        self.integral = 0.0

    def iq_reference(self, speed: float, speed_ref: float, dt: float) -> float:
        error = speed_ref - speed
        self.integral = np.clip(self.integral + error * dt, -8, 8)
        return float(
            np.clip(
                0.25 * error + 1.0 * self.integral,
                -self.params.current_limit_a,
                self.params.current_limit_a,
            )
        )


class FOCPI:
    def __init__(self, params: PMSMParams):
        self.params = params
        self.speed_loop = SpeedReference(params)
        self.integral = np.zeros(2)

    def command(self, state: Array, speed_ref: float, dt: float) -> Array:
        i_d, i_q, omega, _ = state
        p = self.params
        iq_ref = self.speed_loop.iq_reference(omega, speed_ref, dt)
        error = np.array([-i_d, iq_ref - i_q])
        self.integral = np.clip(self.integral + error * dt, -20, 20)
        omega_e = p.pole_pairs * omega
        decoupling = np.array([-omega_e * p.lq_h * i_q, omega_e * (p.ld_h * i_d + p.flux_wb)])
        voltage = 18.0 * error + 900.0 * self.integral + decoupling
        return saturate_vector(voltage, p.phase_voltage_limit)


class PredictiveCurrentControl:
    """One-step deadbeat predictive current controller with voltage-disk projection."""

    def __init__(self, params: PMSMParams):
        self.params = params
        self.speed_loop = SpeedReference(params)

    def command(self, state: Array, speed_ref: float, dt: float) -> Array:
        i_d, i_q, omega, _ = state
        p = self.params
        iq_ref = self.speed_loop.iq_reference(omega, speed_ref, dt)
        omega_e = p.pole_pairs * omega
        v_d = p.resistance_ohm * i_d - omega_e * p.lq_h * i_q + p.ld_h * (0.0 - i_d) / dt
        v_q = (
            p.resistance_ohm * i_q + omega_e * (p.ld_h * i_d + p.flux_wb) + p.lq_h * (iq_ref - i_q) / dt
        )
        return saturate_vector(np.array([v_d, v_q]), p.phase_voltage_limit)


def saturate_vector(vector: Array, limit: float) -> Array:
    norm = np.linalg.norm(vector)
    return np.asarray(vector, dtype=float) * min(1.0, limit / max(norm, 1e-12))


def run_episode(
    controller_name: str = "predictive",
    duration_s: float = 1.0,
    sample_time_s: float = 0.0002,
    resistance_scale: float = 1.0,
    flux_scale: float = 1.0,
) -> dict:
    nominal = PMSMParams()
    plant = PMSM(nominal.drifted(resistance_scale, flux_scale))
    controllers = {"foc-pi": FOCPI(nominal), "predictive": PredictiveCurrentControl(nominal)}
    controller = controllers[controller_name]
    time = np.arange(0, duration_s + sample_time_s / 2, sample_time_s)
    states = np.zeros((len(time), 4))
    voltages = np.zeros((len(time), 2))
    loads = np.zeros(len(time))
    speed_ref = np.where(time < 0.1, 0.0, 100.0)
    for index, now in enumerate(time[:-1]):
        loads[index] = 0.0 if now < 0.5 else 1.8
        voltages[index] = controller.command(states[index], speed_ref[index], sample_time_s)
        states[index + 1] = plant.step(states[index], voltages[index], loads[index], sample_time_s)
    loads[-1] = loads[-2]
    voltages[-1] = voltages[-2]
    evaluation = time >= 0.2
    error = states[:, 2] - speed_ref
    copper_loss_j = float(
        np.trapezoid(1.5 * plant.params.resistance_ohm * np.sum(states[:, :2] ** 2, axis=1), time)
    )
    mechanical_energy_j = (
        float(np.trapezoid(np.maximum(plant.torque(s) * s[2] for s in states), time))
        if False
        else float(np.trapezoid(np.array([max(plant.torque(s) * s[2], 0.0) for s in states]), time))
    )
    return {
        "time": time,
        "states": states,
        "voltages": voltages,
        "loads": loads,
        "speed_reference": speed_ref,
        "speed_rmse_radps": float(np.sqrt(np.mean(error[evaluation] ** 2))),
        "speed_rmse_percent_rated": float(np.sqrt(np.mean(error[evaluation] ** 2)) / 100 * 100),
        "peak_current_a": float(np.max(np.linalg.norm(states[:, :2], axis=1))),
        "peak_voltage_v": float(np.max(np.linalg.norm(voltages, axis=1))),
        "voltage_limit_violations": int(
            np.count_nonzero(np.linalg.norm(voltages, axis=1) > nominal.phase_voltage_limit + 1e-9)
        ),
        "copper_loss_j": copper_loss_j,
        "mechanical_energy_j": mechanical_energy_j,
    }
