from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Aircraft:
    mass: float = 5.0
    gravity: float = 9.81
    rho: float = 1.225
    wing_area: float = 0.8
    mean_chord: float = 0.35
    pitch_inertia: float = 1.0
    cl0: float = 0.20
    cl_alpha: float = 5.0
    cd0: float = 0.050
    induced_drag_factor: float = 0.060
    cm0: float = 0.020
    cm_alpha: float = -0.35
    cm_q: float = -3.5
    cm_delta_e: float = -1.10
    elevator_limit: float = math.radians(17.2)


@dataclass(frozen=True)
class Controller:
    kp: float
    kq: float


@dataclass(frozen=True)
class TrimCondition:
    speed: float
    alpha: float
    theta: float
    elevator: float
    thrust: float


def aero_coefficients(alpha: float, aircraft: Aircraft) -> tuple[float, float]:
    cl = aircraft.cl0 + aircraft.cl_alpha * alpha
    cd = aircraft.cd0 + aircraft.induced_drag_factor * cl**2
    return cl, cd


def solve_level_trim(speed: float, aircraft: Aircraft) -> TrimCondition:
    """Solve steady, level flight with thrust aligned to body pitch angle."""
    dynamic_pressure = 0.5 * aircraft.rho * speed**2
    alpha = (aircraft.mass * aircraft.gravity / (dynamic_pressure * aircraft.wing_area) - aircraft.cl0) / aircraft.cl_alpha

    for _ in range(30):
        cl, cd = aero_coefficients(alpha, aircraft)
        lift = dynamic_pressure * aircraft.wing_area * cl
        drag = dynamic_pressure * aircraft.wing_area * cd
        residual = lift + drag * math.tan(alpha) - aircraft.mass * aircraft.gravity
        epsilon = 1e-7
        cl_plus, cd_plus = aero_coefficients(alpha + epsilon, aircraft)
        lift_plus = dynamic_pressure * aircraft.wing_area * cl_plus
        drag_plus = dynamic_pressure * aircraft.wing_area * cd_plus
        residual_plus = lift_plus + drag_plus * math.tan(alpha + epsilon) - aircraft.mass * aircraft.gravity
        derivative = (residual_plus - residual) / epsilon
        alpha -= residual / derivative

    cl, cd = aero_coefficients(alpha, aircraft)
    drag = dynamic_pressure * aircraft.wing_area * cd
    thrust = drag / math.cos(alpha)
    elevator = -(aircraft.cm0 + aircraft.cm_alpha * alpha) / aircraft.cm_delta_e
    if abs(elevator) > aircraft.elevator_limit:
        raise ValueError("Trim elevator exceeds actuator limit")
    return TrimCondition(speed=speed, alpha=alpha, theta=alpha, elevator=elevator, thrust=thrust)


def elevator_command(
    state: np.ndarray,
    controller: Controller | None,
    trim: TrimCondition,
    aircraft: Aircraft,
) -> float:
    _, _, vx, vz, theta, pitch_rate = state
    gamma = math.atan2(vz, vx)
    alpha = theta - gamma
    if controller is None:
        command = trim.elevator
    else:
        command = (
            trim.elevator
            + controller.kp * (alpha - trim.alpha)
            + controller.kq * pitch_rate
        )
    return float(np.clip(command, -aircraft.elevator_limit, aircraft.elevator_limit))


def state_derivative(
    state: np.ndarray,
    controller: Controller | None,
    trim: TrimCondition,
    aircraft: Aircraft,
) -> np.ndarray:
    x, altitude, vx, vz, theta, pitch_rate = state
    del x, altitude
    speed = max(math.hypot(vx, vz), 1e-6)
    gamma = math.atan2(vz, vx)
    alpha = theta - gamma
    dynamic_pressure = 0.5 * aircraft.rho * speed**2
    cl, cd = aero_coefficients(alpha, aircraft)
    lift = dynamic_pressure * aircraft.wing_area * cl
    drag = dynamic_pressure * aircraft.wing_area * cd
    elevator = elevator_command(state, controller, trim, aircraft)

    aerodynamic_x = -drag * math.cos(gamma) - lift * math.sin(gamma)
    aerodynamic_z = -drag * math.sin(gamma) + lift * math.cos(gamma)
    thrust_x = trim.thrust * math.cos(theta)
    thrust_z = trim.thrust * math.sin(theta)
    ax = (aerodynamic_x + thrust_x) / aircraft.mass
    az = (aerodynamic_z + thrust_z) / aircraft.mass - aircraft.gravity

    reduced_pitch_rate = pitch_rate * aircraft.mean_chord / (2.0 * speed)
    cm = (
        aircraft.cm0
        + aircraft.cm_alpha * alpha
        + aircraft.cm_q * reduced_pitch_rate
        + aircraft.cm_delta_e * elevator
    )
    pitch_moment = dynamic_pressure * aircraft.wing_area * aircraft.mean_chord * cm
    pitch_acceleration = pitch_moment / aircraft.pitch_inertia
    return np.array(
        [vx, vz, ax, az, pitch_rate, pitch_acceleration],
        dtype=float,
    )


def rk4_step(
    state: np.ndarray,
    dt: float,
    controller: Controller | None,
    trim: TrimCondition,
    aircraft: Aircraft,
) -> np.ndarray:
    derivative = lambda current: state_derivative(current, controller, trim, aircraft)
    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt * k1)
    k3 = derivative(state + 0.5 * dt * k2)
    k4 = derivative(state + dt * k3)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def simulate(
    controller: Controller | None,
    trim: TrimCondition,
    aircraft: Aircraft,
    duration: float = 10.0,
    dt: float = 0.01,
    disturbance_time: float = 1.0,
    pitch_rate_disturbance: float = 0.05,
) -> dict[str, np.ndarray]:
    steps = int(round(duration / dt)) + 1
    time = np.linspace(0.0, duration, steps)
    state = np.array([0.0, 0.0, trim.speed, 0.0, trim.theta, 0.0], dtype=float)
    histories = {
        "time_s": time,
        "x_m": np.zeros(steps),
        "altitude_m": np.zeros(steps),
        "vx_mps": np.zeros(steps),
        "vz_mps": np.zeros(steps),
        "theta_rad": np.zeros(steps),
        "pitch_rate_radps": np.zeros(steps),
        "alpha_rad": np.zeros(steps),
        "elevator_rad": np.zeros(steps),
    }
    disturbed = False

    for index, current_time in enumerate(time):
        if not disturbed and current_time >= disturbance_time:
            state[5] += pitch_rate_disturbance
            disturbed = True

        gamma = math.atan2(state[3], state[2])
        histories["x_m"][index] = state[0]
        histories["altitude_m"][index] = state[1]
        histories["vx_mps"][index] = state[2]
        histories["vz_mps"][index] = state[3]
        histories["theta_rad"][index] = state[4]
        histories["pitch_rate_radps"][index] = state[5]
        histories["alpha_rad"][index] = state[4] - gamma
        histories["elevator_rad"][index] = elevator_command(state, controller, trim, aircraft)

        if index < steps - 1:
            state = rk4_step(state, dt, controller, trim, aircraft)

    return histories


def response_metrics(
    result: dict[str, np.ndarray],
    trim: TrimCondition,
    disturbance_time: float = 1.0,
    settling_fraction: float = 0.02,
) -> dict[str, float | None]:
    time = result["time_s"]
    mask = time >= disturbance_time
    post_time = time[mask]
    alpha_error = result["alpha_rad"][mask] - trim.alpha
    elevator_delta = result["elevator_rad"][mask] - trim.elevator
    tolerance = settling_fraction * abs(trim.alpha)

    settling_time = None
    for index in range(len(post_time)):
        if np.all(np.abs(alpha_error[index:]) <= tolerance):
            settling_time = float(post_time[index] - disturbance_time)
            break

    return {
        "peak_abs_aoa_error_deg": float(np.degrees(np.max(np.abs(alpha_error)))),
        "rms_aoa_error_deg": float(np.degrees(np.sqrt(np.mean(alpha_error**2)))),
        "final_aoa_error_deg": float(np.degrees(alpha_error[-1])),
        "settling_time_2pct_s": settling_time,
        "peak_elevator_delta_deg": float(np.degrees(np.max(np.abs(elevator_delta)))),
        "peak_pitch_rate_degps": float(np.degrees(np.max(np.abs(result["pitch_rate_radps"][mask])))),
        "final_altitude_change_m": float(result["altitude_m"][-1] - result["altitude_m"][0]),
    }


def tune_controller(trim: TrimCondition, aircraft: Aircraft) -> tuple[Controller, list[dict[str, float]]]:
    candidates: list[dict[str, float]] = []
    best: tuple[float, Controller] | None = None

    def evaluate_grid(kp_values: np.ndarray, kq_values: np.ndarray) -> None:
        nonlocal best
        for kp in kp_values:
            for kq in kq_values:
                controller = Controller(kp=float(kp), kq=float(kq))
                result = simulate(controller, trim, aircraft, duration=8.0, dt=0.02)
                metrics = response_metrics(result, trim)
                settling = metrics["settling_time_2pct_s"]
                settling_penalty = 10.0 if settling is None else float(settling)
                effort = float(metrics["peak_elevator_delta_deg"])
                score = (
                    1.2 * settling_penalty
                    + 1.5 * float(metrics["rms_aoa_error_deg"])
                    + 0.8 * float(metrics["peak_abs_aoa_error_deg"])
                    + 0.03 * effort
                )
                candidate = {
                    "kp": float(kp),
                    "kq": float(kq),
                    "score": score,
                    **{key: value for key, value in metrics.items() if value is not None},
                }
                candidates.append(candidate)
                if best is None or score < best[0]:
                    best = (score, controller)

    evaluate_grid(np.linspace(0.0, 1.5, 9), np.linspace(0.0, 0.6, 9))
    assert best is not None
    coarse_best = best[1]
    evaluate_grid(
        np.linspace(max(0.0, coarse_best.kp - 0.20), min(1.5, coarse_best.kp + 0.20), 7),
        np.linspace(max(0.0, coarse_best.kq - 0.10), min(0.6, coarse_best.kq + 0.10), 7),
    )
    assert best is not None
    return best[1], candidates


def numerical_jacobian(
    controller: Controller,
    trim: TrimCondition,
    aircraft: Aircraft,
) -> np.ndarray:
    equilibrium = np.array([0.0, 0.0, trim.speed, 0.0, trim.theta, 0.0])
    dynamic_indices = [2, 3, 4, 5]
    jacobian = np.zeros((len(dynamic_indices), len(dynamic_indices)))
    for column, state_index in enumerate(dynamic_indices):
        epsilon = 1e-5
        plus = equilibrium.copy()
        minus = equilibrium.copy()
        plus[state_index] += epsilon
        minus[state_index] -= epsilon
        derivative = (
            state_derivative(plus, controller, trim, aircraft)
            - state_derivative(minus, controller, trim, aircraft)
        ) / (2.0 * epsilon)
        jacobian[:, column] = derivative[dynamic_indices]
    return jacobian


def robustness_study(controller: Controller, aircraft: Aircraft) -> dict[str, object]:
    cases: list[dict[str, float | None]] = []
    for speed in (12.0, 15.0, 18.0):
        case_trim = solve_level_trim(speed, aircraft)
        for disturbance in (0.025, 0.05, 0.10):
            result = simulate(
                controller,
                case_trim,
                aircraft,
                pitch_rate_disturbance=disturbance,
            )
            cases.append(
                {
                    "trim_speed_mps": speed,
                    "disturbance_radps": disturbance,
                    **response_metrics(result, case_trim),
                }
            )
    settling_values = [
        float(case["settling_time_2pct_s"])
        for case in cases
        if case["settling_time_2pct_s"] is not None
    ]
    return {
        "number_of_cases": len(cases),
        "all_cases_settled": len(settling_values) == len(cases),
        "worst_peak_abs_aoa_error_deg": max(float(case["peak_abs_aoa_error_deg"]) for case in cases),
        "worst_rms_aoa_error_deg": max(float(case["rms_aoa_error_deg"]) for case in cases),
        "worst_settling_time_2pct_s": max(settling_values) if settling_values else None,
        "maximum_elevator_delta_deg": max(float(case["peak_elevator_delta_deg"]) for case in cases),
        "cases": cases,
    }


def convergence_study(
    controller: Controller,
    trim: TrimCondition,
    aircraft: Aircraft,
) -> dict[str, object]:
    cases: list[dict[str, float | None]] = []
    for dt in (0.02, 0.01, 0.005):
        result = simulate(controller, trim, aircraft, dt=dt)
        cases.append({"time_step_s": dt, **response_metrics(result, trim)})
    reference = cases[-1]
    coarsest = cases[0]
    peak_difference = abs(
        float(coarsest["peak_abs_aoa_error_deg"]) - float(reference["peak_abs_aoa_error_deg"])
    ) / float(reference["peak_abs_aoa_error_deg"]) * 100.0
    rms_difference = abs(
        float(coarsest["rms_aoa_error_deg"]) - float(reference["rms_aoa_error_deg"])
    ) / float(reference["rms_aoa_error_deg"]) * 100.0
    return {
        "cases": cases,
        "coarse_to_fine_peak_error_difference_percent": peak_difference,
        "coarse_to_fine_rms_error_difference_percent": rms_difference,
    }


def write_csv(path: Path, results: dict[str, dict[str, np.ndarray]]) -> None:
    names = list(results)
    fields = list(results[names[0]])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", *fields])
        for name, result in results.items():
            for row in zip(*(result[field] for field in fields)):
                writer.writerow([name, *row])


def make_plots(path: Path, results: dict[str, dict[str, np.ndarray]], trim: TrimCondition) -> None:
    """Render the comparison without requiring a plotting package."""
    width, height = 1800, 1400
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("arial.ttf", 38)
        label_font = ImageFont.truetype("arial.ttf", 27)
        tick_font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        title_font = label_font = tick_font = ImageFont.load_default()

    draw.text(
        (width // 2, 25),
        "Longitudinal Response to a 0.05 rad/s Pitch-Rate Disturbance",
        fill="black",
        font=title_font,
        anchor="ma",
    )
    plot_specs = [
        ("alpha_rad", "Angle of attack (deg)", np.degrees(trim.alpha)),
        ("pitch_rate_radps", "Pitch rate (deg/s)", None),
        ("elevator_rad", "Elevator (deg)", None),
    ]
    colors = {"open_loop": "#D55E00", "closed_loop": "#0072B2"}
    panel_left, panel_right = 190, width - 70
    panel_height, panel_gap, first_top = 330, 70, 120

    for panel_index, (field, y_label, reference) in enumerate(plot_specs):
        top = first_top + panel_index * (panel_height + panel_gap)
        bottom = top + panel_height
        values = [np.degrees(results[name][field]) for name in results]
        y_min = min(float(np.min(value)) for value in values)
        y_max = max(float(np.max(value)) for value in values)
        if reference is not None:
            y_min = min(y_min, reference)
            y_max = max(y_max, reference)
        padding = max(0.15 * (y_max - y_min), 0.05)
        y_min -= padding
        y_max += padding

        def map_x(value: float) -> float:
            return panel_left + value / 10.0 * (panel_right - panel_left)

        def map_y(value: float) -> float:
            return bottom - (value - y_min) / (y_max - y_min) * panel_height

        draw.rectangle((panel_left, top, panel_right, bottom), outline="black", width=2)
        for tick in range(0, 11, 2):
            x_position = map_x(float(tick))
            draw.line((x_position, top, x_position, bottom), fill="#DDDDDD", width=1)
            if panel_index == 2:
                draw.text((x_position, bottom + 10), str(tick), fill="black", font=tick_font, anchor="ma")
        for tick_index in range(5):
            value = y_min + tick_index * (y_max - y_min) / 4.0
            y_position = map_y(value)
            draw.line((panel_left, y_position, panel_right, y_position), fill="#DDDDDD", width=1)
            draw.text(
                (panel_left - 12, y_position),
                f"{value:.2f}",
                fill="black",
                font=tick_font,
                anchor="rm",
            )
        disturbance_x = map_x(1.0)
        draw.line((disturbance_x, top, disturbance_x, bottom), fill="#777777", width=3)
        if reference is not None:
            reference_y = map_y(reference)
            for x_position in range(panel_left, panel_right, 18):
                draw.line((x_position, reference_y, min(x_position + 10, panel_right), reference_y), fill="black", width=2)

        for name, result in results.items():
            times = result["time_s"]
            degrees = np.degrees(result[field])
            points = [(map_x(float(time)), map_y(float(value))) for time, value in zip(times, degrees)]
            draw.line(points, fill=colors[name], width=4)

        label_box = draw.textbbox((0, 0), y_label, font=label_font)
        label_width = label_box[2] - label_box[0] + 12
        label_height = label_box[3] - label_box[1] + 12
        label_image = Image.new("RGBA", (label_width, label_height), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label_image)
        label_draw.text((6, 4), y_label, fill="black", font=label_font)
        rotated_label = label_image.rotate(90, expand=True)
        image.paste(
            rotated_label,
            (28, int((top + bottom - rotated_label.height) / 2)),
            rotated_label,
        )
        if panel_index == 0:
            draw.line((panel_right - 320, top + 25, panel_right - 250, top + 25), fill=colors["open_loop"], width=5)
            draw.text((panel_right - 235, top + 25), "Open loop", fill="black", font=tick_font, anchor="lm")
            draw.line((panel_right - 320, top + 60, panel_right - 250, top + 60), fill=colors["closed_loop"], width=5)
            draw.text((panel_right - 235, top + 60), "Closed loop", fill="black", font=tick_font, anchor="lm")

    draw.text((width // 2, height - 35), "Time (s)", fill="black", font=label_font, anchor="ma")
    image.save(path)


def main() -> None:
    output_directory = Path(__file__).resolve().parent / "flight_control_results"
    output_directory.mkdir(parents=True, exist_ok=True)
    aircraft = Aircraft()
    trim = solve_level_trim(15.0, aircraft)
    controller, candidates = tune_controller(trim, aircraft)
    results = {
        "open_loop": simulate(None, trim, aircraft),
        "closed_loop": simulate(controller, trim, aircraft),
    }
    metrics = {name: response_metrics(result, trim) for name, result in results.items()}
    eigenvalues = np.linalg.eigvals(numerical_jacobian(controller, trim, aircraft))
    peak_error_reduction = (
        1.0
        - float(metrics["closed_loop"]["peak_abs_aoa_error_deg"])
        / float(metrics["open_loop"]["peak_abs_aoa_error_deg"])
    ) * 100.0
    rms_error_reduction = (
        1.0
        - float(metrics["closed_loop"]["rms_aoa_error_deg"])
        / float(metrics["open_loop"]["rms_aoa_error_deg"])
    ) * 100.0
    settling_time_reduction = (
        1.0
        - float(metrics["closed_loop"]["settling_time_2pct_s"])
        / float(metrics["open_loop"]["settling_time_2pct_s"])
    ) * 100.0
    summary = {
        "model": {
            "integrator": "fourth-order Runge-Kutta",
            "duration_s": 10.0,
            "time_step_s": 0.01,
            "integration_steps": 1000,
            "disturbance_time_s": 1.0,
            "pitch_rate_disturbance_radps": 0.05,
            "gain_combinations_evaluated": len(candidates),
        },
        "aircraft": asdict(aircraft),
        "trim": asdict(trim),
        "selected_controller": asdict(controller),
        "metrics": metrics,
        "improvements": {
            "peak_aoa_error_reduction_percent": peak_error_reduction,
            "rms_aoa_error_reduction_percent": rms_error_reduction,
            "settling_time_reduction_percent": settling_time_reduction,
        },
        "robustness_study": robustness_study(controller, aircraft),
        "time_step_convergence": convergence_study(controller, trim, aircraft),
        "closed_loop_eigenvalues": [
            {"real": float(value.real), "imaginary": float(value.imag)} for value in eigenvalues
        ],
        "closed_loop_max_eigenvalue_real_part": float(max(value.real for value in eigenvalues)),
    }
    with (output_directory / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_csv(output_directory / "time_history.csv", results)
    make_plots(output_directory / "response_comparison.png", results, trim)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

print(f'tolerance = {tolerance}')
print(f'overshoot = {overshoot}')
