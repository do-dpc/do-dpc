# mypy: ignore-errors
# pylint: skip-file
from dataclasses import dataclass
from enum import Enum
from math import pi
from typing import Optional, Tuple


class State(Enum):
    """State definition"""

    x = 0
    y = 1
    x_dot = 2
    y_dot = 3
    theta = 4
    theta_dot = 5
    left_ground_contact = 6
    right_ground_contact = 7


@dataclass
class UserArgs:
    """User arguments for tweaking the environment"""

    render_landing_position: bool = True
    render_lander_center_position: bool = True

    random_initial_position: bool = False
    initial_position: Optional[Tuple[float, ...]] = None  # 3-tuple (x, y, theta)
    initial_state: Optional[Tuple[float, ...]] = None  # 6-tuple (x, y, x_dot, y_dot, theta, theta_dot)

    enable_wind: bool = False
    enable_moving_barge: bool = False

    initial_barge_position: Optional[Tuple[float, ...]] = None  # 2-tuple (x, theta)


@dataclass(frozen=True)
class EnvConfig:
    """
    Notes about the Falcon 9:
    - Total thrust of the first stage is: 6.8 MN
    - Dimensions are approximately 42.7m long and 3.7m in diameter
    - First stage mass on landing: 25,000-30,000 kg
    - Density of original rocket is 158.238 kg / m^2

    """

    ## SIMULATION
    scale: int = 30  # adjusts pixels to units conversions; should keep at 30
    viewport_width: int = 1000
    viewport_height: int = 800

    width = viewport_width / scale
    height = viewport_height / scale

    sea_chunks_x: int = 16  # how many steps are used to draw the sea (x-direction)
    sea_chunks_y: int = 24  # how many steps are used to draw the sea (y-direction)

    clouds: bool = True

    # simulation timestep
    fps: int = 60
    update_time: float = 1 / fps

    ## ROCKET PARAMETERS
    # geometry
    lander_scaling: float = 4  # controls the rocket dimensions

    # length and half-width (radius)
    lander_length: float = 42.7 * lander_scaling
    lander_radius: float = 1.85 * lander_scaling

    # mass and gravity
    lander_density: float = 158.238  # using 25,000 kg at landing
    gravity: float = -9.81

    lander_poly: Tuple[Tuple, ...] = (
        (-lander_radius, -lander_length / 2),
        (+lander_radius, -lander_length / 2),
        (+lander_radius, +lander_length / 2),
        (-lander_radius, +lander_length / 2),
    )

    nozzle_poly: Tuple[Tuple, ...] = (
        (-lander_radius / 2, -lander_length / 32),
        (+lander_radius / 2, -lander_length / 32),
        (-lander_radius / 2, +lander_length / 32),
        (+lander_radius / 2, +lander_length / 32),
    )

    leg_width: float = 1.2 * lander_scaling
    leg_height: float = lander_length / 4  # already scaled to the lander length
    leg_initial_angle: float = 30
    # approximate torque needed to support the lander mass
    # doesn't account for the leg angle, or the fact that there are two legs, want it to be stiff
    leg_spring_torque: float = (lander_length * 2 * lander_radius * lander_density * gravity) * leg_height

    # we offset from the center of the lander
    side_engine_y_offset: float = (lander_length / 2) * 3 / 4  # three quarters up the top half
    side_engine_x_offset: float = lander_radius  # on the side of the lander

    # forces, costs, torque, friction
    # private, actual value
    _main_engine_thrust: float = 6.8e6 * lander_scaling**3  # newtons, some scaling for this sim
    _side_engine_thrust: float = (
        _main_engine_thrust / 50
    )  # can apply in either direction, -100% to 100% available on landing

    # public, scaled value
    main_engine_thrust = _main_engine_thrust / scale**3
    side_engine_thrust = _side_engine_thrust / scale**3

    main_engine_thrust_limits: Tuple[float, ...] = (
        0.0,
        1.0,
    )  # between 0 - 100 % available on landing
    side_engine_thrust_limits: Tuple[float, ...] = (
        -1.0,
        1.0,
    )  # between -100% to 100% available on landing

    initial_fuel_mass: float = 0.2
    main_engine_fuel_cost: float = main_engine_thrust / side_engine_thrust
    side_engine_fuel_cost: float = 1.0

    nozzle_torque: float = 500 * lander_scaling
    max_nozzle_angle: float = 15 * pi / 180
    nozzle_angle_limits: Tuple[float, ...] = (-1.0, 1.0)

    # state reset limits
    theta_limit = 35 * pi / 180

    ## LANDING PARAMETERS
    barge_friction: float = 2.0 * lander_scaling
    barge_width: float = (width / 16) * lander_scaling
    barge_height: float = (height / 64) * lander_scaling
    barge_y_pos: float = height / 8

    landing_vertical_calibration: float = (
        -0.15
    )  # fudge factor to ensure that most controllers will trigger the contact detector

    # disturbances
    wind_force = main_engine_thrust / 50
