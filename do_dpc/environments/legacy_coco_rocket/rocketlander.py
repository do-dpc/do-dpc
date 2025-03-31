# mypy: ignore-errors
# pylint: skip-file
"""
A Box2D environment for Gymnasium which emulates the Falcon 9 barge landing
Original environment created by Reuben Ferrante

Modifications by Dylan Vogel and Gerasimos Maltezos for the 2023 Computational Control course offered at ETH Zurich

"""
import copy
import warnings
from typing import Dict, List, Tuple

import Box2D
import gymnasium as gym
import numpy as np
import pygame
from Box2D.b2 import (
    circleShape,
    contactListener,
    edgeShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
)
from gymnasium import spaces

from do_dpc.environments.legacy_coco_rocket.env_cfg import State, EnvConfig, UserArgs

# for warnings
YELLOW = "\x1b[33;20m"
ENDL = "\x1b[0m"

# for referencing different parts of the state
DEGTORAD = np.pi / 180
XX = State.x.value
YY = State.y.value
X_DOT = State.x_dot.value
Y_DOT = State.y_dot.value
THETA = State.theta.value
THETA_DOT = State.theta_dot.value
LEFT_GROUND_CONTACT = State.left_ground_contact.value
RIGHT_GROUND_CONTACT = State.right_ground_contact.value


## CONTACT DETECTOR


class ContactDetector(contactListener):
    """Callback class for making/braking contact in the environment"""

    def __init__(self, env: gym.Env):
        """Constructor method

        Args:
            env (gym.Env): gym environment to listen on
        """
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        """Called when contact between two bodies begins in the environment"""
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True

        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True
                if self.env.barge not in [contact.fixtureA.body, contact.fixtureB.body]:
                    self.env.game_over = True

    def EndContact(self, contact):
        """Called when contact betwene two bodies ends in the environment"""
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


## GYMNASIUM METHODS


class RocketLander(gym.Env):
    """Gymnasum environment which models a Falcon 9 barge landing"""

    # required by gymnasium
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, args: Dict, render_mode=None):
        """Constructor method"""
        # load the environment configuration
        self.cfg = EnvConfig()
        self.args = UserArgs(**args)

        # environment
        self.world = Box2D.b2World(gravity=(0, self.cfg.gravity))

        # set gymnasium action and observation spaces
        self.action_space = spaces.Box(
            low=np.array(
                [
                    self.cfg.main_engine_thrust_limits[0],
                    self.cfg.side_engine_thrust_limits[0],
                    self.cfg.nozzle_angle_limits[0],
                ]
            ),
            high=np.array(
                [
                    self.cfg.main_engine_thrust_limits[1],
                    self.cfg.side_engine_thrust_limits[1],
                    self.cfg.nozzle_angle_limits[1],
                ]
            ),
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.inf, -np.inf, -self.cfg.theta_limit, -np.inf, 0, 0]),
            high=np.array(
                [
                    self.cfg.width,
                    self.cfg.height,
                    np.inf,
                    np.inf,
                    self.cfg.theta_limit,
                    -np.inf,
                    1,
                    1,
                ]
            ),
            dtype=np.float64,
        )

        # bodies
        self.lander = None
        self.legs = []
        self.nozzle = None
        self.barge = None
        self.particles = []
        self.lander_drawlist = []

        # polygons
        self.sea = None
        self.sea_polys = []
        self.sky_polys = []
        self.clouds = []

        # rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.canvas = None

        # state variables
        self.state = []
        self.previous_state = None
        self.game_over = False
        self.contact_flag = False
        self.prev_shaping = None
        self.wind_idx = None
        self.barge_idx = None

        self.initial_barge_position = None

        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        np.random.seed(seed=seed)

        self._destroy()
        self.world.contactListener = ContactDetector(self)

        # state variables
        self.state = []
        self.game_over = False
        self.contact_flag = False
        self.prev_shaping = None
        self.wind_idx = np.random.randint(low=-9999, high=9999)
        self.barge_idx = np.random.randint(low=-9999, high=9999)

        # rendering
        self.lander_drawlist = []

        # set rocket initial position
        new_state = None

        # NOTE:
        # - Directly setting the position of dynamic box2-py objects results in strange behaviour
        # - We simply create the rocket at the correct initial position
        # - Other state values can be set relatively error-free
        pos_x, pos_y = 0.5 * self.cfg.width, 0.9 * self.cfg.height

        if self.args.initial_state is not None:
            assert len(self.args.initial_state) == 6, "Initial state is of incorrect length, should be length 6"
            if self.args.random_initial_position:
                warnings.warn(YELLOW + "WARN: initial_state is set but the initial position will be randomized" + ENDL)
            if self.args.initial_position:
                warnings.warn(YELLOW + "WARN: ignoring the initial_position setting since initial_state is set" + ENDL)

            pos_x = self.args.initial_state[0] * self.cfg.width
            pos_y = self.args.initial_state[1] * self.cfg.height

            new_state = {
                "x_dot": self.args.initial_state[2],
                "y_dot": self.args.initial_state[3],
                "theta": self.args.initial_state[4],
                "theta_dot": self.args.initial_state[5],
            }

        elif self.args.initial_position is not None:
            assert (
                len(self.args.initial_position) == 3
            ), "Initial position is of incorrect length, should be length 3 (x, y, theta)"
            if self.args.random_initial_position:
                warnings.warn(YELLOW + "WARN: initial_state is set but the initial position will be randomized" + ENDL)

            pos_x = self.args.initial_position[0] * self.cfg.width
            pos_y = self.args.initial_position[1] * self.cfg.height

            new_state = {
                "theta": self.args.initial_position[2],
            }

        if self.args.random_initial_position:
            pos_x = np.random.uniform(0, 1) * self.cfg.width
            pos_y = np.random.uniform(0.4, 1) * self.cfg.height
            new_state = {"theta": np.random.uniform(-1, 1) * (self.cfg.theta_limit / 2)}

        # create dynamic bodies
        self._create_rocket(pos_x, pos_y)
        self.lander_drawlist = self.legs + [self.nozzle] + [self.lander]

        # create environment objects
        sea_y_values, sea_x_values = self._create_sea_height(self.cfg.sea_chunks_x)
        self._create_environment(sea_y_values, sea_x_values)
        self._create_clouds()
        self._create_barge()

        if self.args.initial_barge_position is not None:
            self.barge.position = (
                self.args.initial_barge_position[0] * self.cfg.width,
                self.barge.position[1],
            )
            self.barge.angle = self.args.initial_barge_position[1]

        self.initial_barge_position = list(self.barge.position) + [float(self.barge.angle)]

        if new_state is not None:
            self.adjust_dynamics(self.lander, **new_state)

        obs, _, _, _, info = self.step(np.array([0, 0, 0]))
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Simulate the environment forward by one step

        Args:
            action (np.ndarray): actions to apply

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: observation, reward, done, _, info
        """
        if isinstance(action, list):
            action = np.array(action)

        assert (
            action.shape == self.action_space.shape
        ), f"Incorrect action shape, expected: {self.action_space.shape} but got: {action.shape}"

        # adjust nozzle angle
        nozzle_angle = float(action[2])
        nozzle_angle = (
            np.clip(
                nozzle_angle,
                a_min=self.cfg.nozzle_angle_limits[0],
                a_max=self.cfg.nozzle_angle_limits[1],
            )
            * self.cfg.max_nozzle_angle
        )
        self.nozzle.angle = self.lander.angle + nozzle_angle

        # apply forces
        _ = self.__main_engines_force_computation(action)
        _, _ = self.__side_engines_force_computation(action)

        # apply disturbances
        self.__apply_wind_disturbance()
        self.__apply_barge_disturbance()

        # perform simulation step
        self.previous_state = self.state
        self.state = self.__generate_state()
        self._update_particles()  # delete old particles; aesthetic

        reward = self.__compute_reward(self.state, action)

        # check ground contact
        if (self.legs[0].ground_contact or self.legs[1].ground_contact) and self.contact_flag is False:
            self.contact_flag = True

        # check termination conditions
        # add a small factor in case the controller goes right to the edge
        done = False
        if any(
            [
                self.game_over,
                abs(self.state[THETA]) > self.cfg.theta_limit + 0.1,
                self.state[XX] > self.cfg.width + 0.1,
                self.state[XX] < 0 - 0.1,
                self.state[YY] > self.cfg.height + 0.1,
            ]
        ):
            done = True
            reward = -100

        # check if simulation finished in general
        if not self.lander.awake:
            done = True
            reward = +100

        # render if necessary
        if self.render_mode == "human":
            self._render_frame()

        return (
            np.array(self.state),
            reward,
            done,
            False,
            {},
        )

    def close(self):
        super().close()
        self._destroy()
        if self.window is not None:
            pygame.display.quit()

    ## QUERY FUNCTIONS

    def get_mass_properties(self) -> Tuple[float, float]:
        """Get approximate mass and inertia of rocket body
        We assume that most of the inertia is in the main body (rectangle) and neglect
        contributions of the lander legs and nozzle

        Returns:
            Tuple[float, float]: mass, inertia
        """
        return (
            self.lander.mass + self.nozzle.mass + self.legs[0].mass + self.legs[1].mass,
            self.lander.inertia,
        )

    def get_dimensional_properties(self) -> Tuple[float, float]:
        """Get the l1 and l2 dimensions of the rocket
        - l1: longitudinal distance from rocket center to where the nozzle force is applied
        - l2: longitudinal distance from rocket center to height at which the side engine force is applied

        l1 and l2 are measured along the centerline of the rocket

        Returns:
            Tuple[float, float]: l1, l2
        """
        l1 = (self.cfg.lander_length / 2) / self.cfg.scale
        l2 = (self.cfg.side_engine_y_offset) / self.cfg.scale

        return l1, l2

    def get_landing_position(self) -> Tuple[float, float, float]:
        """Get the landing point at the center of the barge, including tilt

        Returns:
            Tuple[float, float, float]: x, y, theta position of the barge
        """
        assert self.barge is not None, "Please call reset() first!"

        pos = list(self.barge.position)
        angle = copy.deepcopy(self.barge.angle)  # prevent users from changing the barge angle

        pos[0] -= np.sin(angle) * self.cfg.barge_height / 2
        pos[1] += np.cos(angle) * self.cfg.barge_height / 2

        rocket_base_to_center_len = (
            np.cos(self.cfg.leg_initial_angle * DEGTORAD) * (self.cfg.leg_height / 2) + (self.cfg.lander_length / 2)
        ) / self.cfg.scale
        rocket_base_to_center_len += self.cfg.landing_vertical_calibration

        rocket_base_to_center = [
            -np.sin(angle) * rocket_base_to_center_len,
            np.cos(angle) * rocket_base_to_center_len,
        ]

        pos[0] += rocket_base_to_center[0]
        pos[1] += rocket_base_to_center[1]

        pos.append(angle)

        return pos

    ## ENVIRONMENT HELPER FUNCTIONS

    def _destroy(self):
        if self.barge is None:
            # nothing has been created yet
            return

        # clean up environment
        self.world.contactListener = None
        self._clean_particles(True)

        self.world.DestroyBody(self.barge)
        self.barge = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.nozzle)
        self.nozzle = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])
        self.world.DestroyBody(self.sea)
        self.sea = None

        self.lander_drawlist = []

    def __compute_reward(self, state: list, action: np.ndarray) -> float:
        """Generate an environment reward based on the current state and actions
        Our reward is based on the lunar lander continuous reward

        Args:
            state (list): current state
            action (np.ndarray): current action

        Returns:
            float: reward
        """
        landing_pos = self.get_landing_position()
        x_pos_norm = (state[0] - landing_pos[0]) / self.cfg.width
        y_pos_norm = (state[1] - landing_pos[1]) / self.cfg.height

        x_vel_norm = state[2] / self.cfg.width
        y_vel_norm = state[3] / self.cfg.height

        reward = 0
        shaping = (
            -100 * np.sqrt(x_pos_norm**2 + y_pos_norm**2)
            - 10 * np.sqrt(x_vel_norm**2 + y_vel_norm**2)
            - 100 * abs(state[4])
            + 20 * state[6]
            + 20 * state[7]
        )

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= action[0] * 0.30
        reward -= action[1] * 0.03

        return reward

    def __main_engines_force_computation(self, action):
        cmd_engine_thrust = 0

        if action[0] > 0:
            sin = np.sin(self.nozzle.angle)
            cos = np.cos(self.nozzle.angle)

            clipped_action = np.clip(
                action[0],
                a_min=self.cfg.main_engine_thrust_limits[0],
                a_max=self.cfg.main_engine_thrust_limits[1],
            )
            cmd_engine_thrust = clipped_action * self.cfg.main_engine_thrust  # inertial forces scale with the cubic

            # apply at the base of the lander, not on the nozzle
            force_pos = (
                self.lander.position[0] + np.sin(self.lander.angle) * (self.cfg.lander_length / 2) / self.cfg.scale,
                self.lander.position[1] - np.cos(self.lander.angle) * (self.cfg.lander_length / 2) / self.cfg.scale,
            )
            force_vector = (-sin * cmd_engine_thrust, cos * cmd_engine_thrust)

            self.lander.ApplyForce(force_vector, force_pos, True)

            # visual particle effects

            # particle dispersion
            angle_dispersion = self.np_random.uniform(-1.0, 1.0) * 15 * DEGTORAD
            force_dispersion = self.np_random.uniform(0.7, 1.0)

            particle_force_vector = (
                np.sin(self.nozzle.angle + angle_dispersion) * cmd_engine_thrust * force_dispersion,
                -np.cos(self.nozzle.angle + angle_dispersion) * cmd_engine_thrust * force_dispersion,
            )

            # scale with the lander_density
            particle = self._create_particle(
                self.cfg.lander_density
                * np.pi
                * (30 / self.cfg.lander_scaling) ** 2
                / self.cfg.scale**2,  # 30 is a 'look good' factor, chosen qualitatively for the mass scaling
                force_pos[0],
                force_pos[1]
                - np.cos(self.lander.angle)
                * (self.cfg.lander_length / 24)
                / self.cfg.scale,  # offset to bottom of nozzle
                1.5,  # particle life decrements by 0.1 every interval, dies at 0
                radius=1.5 * self.cfg.lander_scaling,
            )

            particle.ApplyForce(particle_force_vector, force_pos, True)

        return cmd_engine_thrust

    def __side_engines_force_computation(self, action):
        cmd_side_engine_thrust = 0
        side_engine_dir = 1

        if np.abs(action[1]) > 0:
            clipped_action = np.clip(
                action[1],
                a_min=self.cfg.side_engine_thrust_limits[0],
                a_max=self.cfg.side_engine_thrust_limits[1],
            )
            cmd_side_engine_thrust = (
                clipped_action * self.cfg.side_engine_thrust
            )  # inertial forces scale with the cubic

            # convention is that Fs = Fl - Fr
            # Therefore if sign is positive we fire the left thruster
            side_engine_dir = np.sign(cmd_side_engine_thrust)
            cmd_side_engine_thrust = np.abs(cmd_side_engine_thrust)

            # shorten some equations
            sin = np.sin(self.lander.angle)
            cos = np.cos(self.lander.angle)

            # apply at side_engine_y_offset up the lander from the center, with some left/right offset
            force_pos = (
                self.lander.position[0]
                + ((-sin * self.cfg.side_engine_y_offset) - (side_engine_dir * cos * self.cfg.side_engine_x_offset))
                / self.cfg.scale,
                self.lander.position[1]
                + ((cos * self.cfg.side_engine_y_offset) - (side_engine_dir * sin * self.cfg.side_engine_x_offset))
                / self.cfg.scale,
            )
            force_vector = (
                side_engine_dir * cos * cmd_side_engine_thrust,
                side_engine_dir * sin * cmd_side_engine_thrust,
            )

            self.lander.ApplyForce(force_vector, force_pos, True)

            # visual particle effects

            # particle dispersion
            angle_dispersion = self.np_random.uniform(-1.0, 1.0) * 10 * DEGTORAD
            force_dispersion = self.np_random.uniform(0.4, 1.0)

            # opposite direction to force applied on the lander
            particle_force_vector = (
                -side_engine_dir
                * np.cos(self.lander.angle + angle_dispersion)
                * cmd_side_engine_thrust
                * force_dispersion,
                -side_engine_dir
                * np.sin(self.lander.angle + angle_dispersion)
                * cmd_side_engine_thrust
                * force_dispersion,
            )

            # scale with the lander_density
            particle = self._create_particle(
                self.cfg.lander_density
                * np.pi
                * (20 / self.cfg.lander_scaling) ** 2
                / self.cfg.scale**2,  # 20 is a 'look good' factor, chosen qualitatively for the mass scaling
                force_pos[0],
                force_pos[1],
                0.5,  # particle 'life' decrements by 0.1 each interval, expires at 0
                radius=0.75 * self.cfg.lander_scaling,
            )
            particle.ApplyForce(particle_force_vector, force_pos, True)

        return cmd_side_engine_thrust, side_engine_dir

    def __generate_state(self):
        """Take one simulation step forward and return the new state

        Returns:
            list: x, y, x_dot, y_dot, theta, theta_dot, left_contact, right_contact
        """
        self.world.Step(1.0 / self.cfg.fps, 6 * 30, 6 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity

        state = [
            pos.x,
            pos.y,
            vel.x,
            vel.y,
            self.lander.angle,
            self.lander.angularVelocity,
            1 if self.legs[0].ground_contact else 0,
            1 if self.legs[1].ground_contact else 0,
        ]

        return state

    def __apply_wind_disturbance(self):
        """Apply a wind force to the lander along the x-axis"""
        if self.args.enable_wind and not (self.legs[0].ground_contact or self.legs[1].ground_contact):
            wind_mag = (
                np.tanh(np.sin(0.02 * self.wind_idx) + (np.sin(np.pi * 0.01 * self.wind_idx))) * self.cfg.wind_force
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter(
                (wind_mag, 0.0),
                True,
            )

    def __apply_barge_disturbance(self):
        """Apply x, y, theta offsets to the barge from the initial position"""
        if self.args.enable_moving_barge:
            x_rate = 0.01
            y_rate = 0.006
            theta_rate = 0.004

            barge_x = (
                np.tanh(np.sin(x_rate * self.barge_idx) + (np.sin(np.pi * (x_rate / 2) * self.barge_idx)))
                * self.cfg.barge_width
                / 16
            ) + self.initial_barge_position[0]

            barge_y = (
                np.tanh(np.sin(y_rate * self.barge_idx) + (np.sin(np.pi * (y_rate / 2) * self.barge_idx)))
                * self.cfg.barge_height
                / 4
            ) + self.initial_barge_position[1]

            barge_theta = (
                np.tanh(np.sin(theta_rate * self.barge_idx) + (np.sin(np.pi * (theta_rate / 2) * self.barge_idx)))
                * 5
                * DEGTORAD
            ) + self.initial_barge_position[2]

            self.barge_idx += 1

            self.barge.position = (barge_x, barge_y)
            self.barge.angle = barge_theta

    ## RENDERING HELPERS

    def _create_rocket(self, pos_x: float, pos_y: float):
        """Create the dynamic rocket object"""

        # variables
        lander_body_color = (255, 255, 255)

        # lander body
        self.lander = self.world.CreateDynamicBody(
            position=(pos_x, pos_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / self.cfg.scale, y / self.cfg.scale) for x, y in self.cfg.lander_poly]
                ),
                density=self.cfg.lander_density,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x0003,  # contact ground and sea
                restitution=0.0,
            ),
        )
        self.lander.color1 = lander_body_color
        self.lander.color2 = (0, 0, 0)

        # lander legs
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                # will move when we attach the revolute joint
                position=(pos_x, pos_y),
                angle=0,
                fixtures=fixtureDef(
                    # box assumes half-width and half-lengths
                    shape=polygonShape(
                        box=(
                            self.cfg.leg_width / (2 * self.cfg.scale),
                            self.cfg.leg_height / (2 * self.cfg.scale),
                        )
                    ),
                    density=self.cfg.lander_density,  # cannot assume massless because for some reason we need density to apply torque?
                    friction=0.1,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x0003,  # contact ground and sea
                ),
            )

            leg.ground_contact = False
            leg.color1 = lander_body_color
            leg.color2 = (0, 0, 0)

            # join legs to lander body
            revolute_joint = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                # basically, put the legs at 1/4 ond 3/4 across the body (radius/2), and align the center
                localAnchorA=(
                    (i * self.cfg.lander_radius / 2) / self.cfg.scale,
                    (-self.cfg.lander_length / 2 + self.cfg.leg_height / 2) / self.cfg.scale,
                ),
                localAnchorB=(0, (self.cfg.leg_height / 2) / self.cfg.scale),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.cfg.leg_spring_torque,
                motorSpeed=1.0 * i,
            )

            # set angle limits
            if i == -1:
                revolute_joint.lowerAngle = -(self.cfg.leg_initial_angle + 2) * DEGTORAD
                revolute_joint.upperAngle = -self.cfg.leg_initial_angle * DEGTORAD
            else:
                revolute_joint.lowerAngle = self.cfg.leg_initial_angle * DEGTORAD
                revolute_joint.upperAngle = (self.cfg.leg_initial_angle + 2) * DEGTORAD
            leg.joint = self.world.CreateJoint(revolute_joint)
            self.legs.append(leg)

        # nozzle
        self.nozzle = self.world.CreateDynamicBody(
            position=(pos_x, pos_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / self.cfg.scale, y / self.cfg.scale) for x, y in self.cfg.nozzle_poly]
                ),
                density=self.cfg.lander_density,
                friction=0.1,
                categoryBits=0x0040,
                maskBits=0x0003,  # contact sea and ground
                restitution=0.0,
            ),
        )

        self.nozzle.color1 = (0, 0, 0)
        self.nozzle.color2 = (0, 0, 0)

        # join nozzle to lander body
        revolute_joint = revoluteJointDef(
            bodyA=self.lander,
            bodyB=self.nozzle,
            localAnchorA=(0, -(self.cfg.lander_length / 2) / self.cfg.scale),
            localAnchorB=(0, 0),
            enableMotor=True,
            enableLimit=False,
            maxMotorTorque=self.cfg.nozzle_torque,
            motorSpeed=0,
            referenceAngle=0,
            lowerAngle=self.cfg.max_nozzle_angle * self.cfg.nozzle_angle_limits[0],
            upperAngle=self.cfg.max_nozzle_angle * self.cfg.nozzle_angle_limits[1],
        )

        self.nozzle.joint = self.world.CreateJoint(revolute_joint)

        return

    def _create_barge(self):
        """Create the landing barge"""
        # start it in the center of the screen
        barge_center = (0.5 * self.cfg.width, self.cfg.barge_y_pos)

        self.barge = self.world.CreateStaticBody(
            position=barge_center,
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(box=(self.cfg.barge_width / 2, self.cfg.barge_height / 2)),
                categoryBits=0x0001,
                maskBits=0xFFFF,
                friction=self.cfg.barge_friction,
            ),
        )
        self.barge.color1 = (40, 40, 40)
        self.barge.color2 = (25, 25, 25)

    def _create_sea_height(self, chunks: int) -> Tuple[list, list]:
        """Create the sea height

        Args:
            chunks (int): number of chunks to divide the sea into

        Returns:
            _type_: _description_
        """

        # ocean height
        sea_height = np.random.normal(self.cfg.barge_y_pos, 0.5, size=(chunks + 1,))
        sea_x = [self.cfg.width / (chunks - 1) * i for i in range(chunks)]

        # set the height of the sea near the barge to be the barge height
        sea_height[chunks // 2 - 2] = self.cfg.barge_y_pos
        sea_height[chunks // 2 - 1] = self.cfg.barge_y_pos
        sea_height[chunks // 2 + 0] = self.cfg.barge_y_pos
        sea_height[chunks // 2 + 1] = self.cfg.barge_y_pos
        sea_height[chunks // 2 + 2] = self.cfg.barge_y_pos

        smoothed_sea_height = [
            0.33 * (sea_height[i - 1] + sea_height[i + 0] + sea_height[i + 1]) for i in range(chunks)
        ]

        return smoothed_sea_height, sea_x

    def _create_environment(self, sea_y_values, sea_x_values):
        """Create the environment polygons"""
        assert len(sea_y_values) == self.cfg.sea_chunks_x and len(sea_x_values) == self.cfg.sea_chunks_x
        num_segments = self.cfg.sea_chunks_x

        self.sea = self.world.CreateStaticBody(
            # create bottom edge of the sea
            shapes=edgeShape(vertices=[(0, 0), (self.cfg.width, 0)])
        )
        self.sky_polys = []
        self.sea_polys = [[] for _ in range(self.cfg.sea_chunks_y)]  # purely aesthetic

        for i in range(num_segments - 1):
            # p1 and p2 define the top edge of the sea chunk
            p1 = (sea_x_values[i], sea_y_values[i])
            p2 = (sea_x_values[i + 1], sea_y_values[i + 1])

            # create collisions, add sky
            self.sea.CreateEdgeFixture(
                vertices=[p1, p2],
                density=0,
                friction=0.1,
                categoryBits=0x0002,
                maskBits=0x00FF,
            )  # create sea collisions
            self.sky_polys.append([p1, p2, (p2[0], self.cfg.height), (p1[0], self.cfg.height)])  # draw sky above sea

            # make sea aesthetic by adding gradient polygons
            for j in range(self.cfg.sea_chunks_y):
                # draw from top to bottom
                k = 1 - j / self.cfg.sea_chunks_y
                self.sea_polys[j].append([(p1[0], p1[1] * k), (p2[0], p2[1] * k), (p2[0], 0), (p1[0], 0)])

    def _create_clouds(self):
        self.clouds = []
        for _ in range(10):
            self.clouds.append(self._create_cloud([0.2, 0.4], [0.65, 0.7], 1))
            self.clouds.append(self._create_cloud([0.7, 0.8], [0.75, 0.8], 1))

    def _create_cloud(self, x_range, y_range, y_variance=0.1):
        cloud_poly = []
        numberofdiscretepoints = 3

        initial_y = (self.cfg.viewport_height * np.random.uniform(y_range[0], y_range[1])) / self.cfg.scale
        initial_x = (self.cfg.viewport_width * np.random.uniform(x_range[0], x_range[1])) / self.cfg.scale

        y_coordinates = np.random.normal(0, y_variance, numberofdiscretepoints)
        x_step = np.linspace(initial_x, initial_x + np.random.uniform(1, 6), numberofdiscretepoints + 1)

        for i in range(0, numberofdiscretepoints):
            cloud_poly.append((x_step[i], initial_y + np.sin(3.14 * 2 * i / 50) * y_coordinates[i]))

        return cloud_poly

    def _create_particle(self, mass, x, y, ttl, radius=3):
        """
        Used for both the Main Engine and Side Engines
        :param mass: Different mass to represent different forces
        :param x: x position
        :param y: y position
        :param ttl:
        :param radius:
        :return:
        """
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=radius / self.cfg.scale, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x0001,  # collide only with ground, not with the sea
                restitution=0.3,
            ),
        )
        p.ttl = ttl  # ttl is decreased with every time step to determine if the particle should be destroyed
        self.particles.append(p)
        # Check if some particles need cleaning
        self._clean_particles(False)
        return p

    def _update_particles(self):
        for obj in self.particles:
            obj.ttl -= 0.1
            color_red = min(255, max(50, 50 + 255 * obj.ttl))
            color_green_blue = min(255, max(50, 0.5 * 255 * obj.ttl))
            obj.color1 = self._cast_tuple_to_int((color_red, color_green_blue, color_green_blue))
            obj.color2 = self._cast_tuple_to_int((color_red, color_green_blue, color_green_blue))

        self._clean_particles(False)

    def _clean_particles(self, all_particles):
        while self.particles and (all_particles or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """Render a single frame"""
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((self.cfg.viewport_width, self.cfg.viewport_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self.canvas = pygame.Surface((self.cfg.viewport_width, self.cfg.viewport_height))
        pygame.transform.scale(self.canvas, (self.cfg.scale, self.cfg.scale))
        self.canvas.fill((255, 255, 255))

        self._render_sky()
        self._render_lander()
        self._render_environment()

        if self.args.render_lander_center_position:
            self._draw_marker(
                x=self.lander.position.x,
                y=self.lander.position.y,
                theta=self.lander.angle,
                color=(0, 0, 0),
            )
        if self.args.render_landing_position:
            landing_pos = self.get_landing_position()
            self._draw_marker(
                x=landing_pos[0],
                y=landing_pos[1],
                theta=landing_pos[2],
                color=(11, 218, 81),
            )

        self.canvas = pygame.transform.flip(self.canvas, False, True)

        if self.render_mode == "human":
            # copies our drawings from canvas to visible window
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # to ensure that human rendering occurs at a predefined framerate
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))

    def _render_lander(self):
        for obj in self.particles + self.lander_drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.canvas,
                        color=obj.color1,
                        center=trans * f.shape.pos * self.cfg.scale,
                        radius=f.shape.radius * self.cfg.scale,
                    )
                    pygame.draw.circle(
                        self.canvas,
                        color=obj.color2,
                        center=trans * f.shape.pos * self.cfg.scale,
                        radius=f.shape.radius * self.cfg.scale,
                    )
                else:
                    # Lander
                    path = [trans * v for v in f.shape.vertices]

                    pygame.draw.polygon(
                        self.canvas,
                        color=obj.color1,
                        points=self._scale_list_of_tuples(path, self.cfg.scale),
                    )
                    pygame.draw.aalines(
                        self.canvas,
                        color=obj.color2,
                        points=self._scale_list_of_tuples(path, self.cfg.scale),
                        closed=True,
                    )

    def _render_sky(self):
        for p in self.sky_polys:
            pygame.draw.polygon(
                self.canvas,
                color=(212, 234, 255),
                points=self._scale_list_of_tuples(p, self.cfg.scale),
            )

        if self.cfg.clouds:
            for x in self.clouds:
                pygame.draw.polygon(
                    self.canvas,
                    color=(255, 255, 255),
                    points=self._scale_list_of_tuples(x, self.cfg.scale),
                )

    def _render_environment(self):
        """Render the sea and barge"""
        for i, s in enumerate(self.sea_polys):
            k = 1 - (i + 1) / self.cfg.sea_chunks_y
            for poly in s:
                pygame.draw.polygon(
                    self.canvas,
                    color=self._cast_tuple_to_int((0, 0.5 * 255 * k, min(255, 255 * k + 128))),
                    points=self._scale_list_of_tuples(poly, self.cfg.scale),
                )

        # barge
        for f in self.barge.fixtures:
            trans = f.body.transform

            path = [trans * v for v in f.shape.vertices]

            pygame.draw.polygon(
                self.canvas,
                color=self.barge.color1,
                points=self._scale_list_of_tuples(path, self.cfg.scale),
            )
            pygame.draw.aalines(
                self.canvas,
                color=self.barge.color2,
                points=self._scale_list_of_tuples(path, self.cfg.scale),
                closed=True,
            )

        # landing flags
        trans = self.barge.fixtures[0].body.transform  # should only be one

        for ind in [-1, +1]:
            # set some x and y points for the flag
            flag_x = ind * (3 / 4) * self.cfg.barge_width / 2
            flag_y = self.cfg.barge_height / 2

            # these units are meters
            flag_pole_length = 0.8
            flag_triangle_height = 0.3
            flag_triangle_width = 0.8

            flag_triangle = [
                (flag_x, flag_y + flag_pole_length),
                (flag_x, flag_y + flag_pole_length + flag_triangle_height),
                (
                    flag_x + flag_triangle_width,
                    flag_y + flag_pole_length + flag_triangle_height / 2,
                ),
            ]

            # flag_triangle = polygonShape(vertices=flag_triangle)
            flag_triangle = [trans * v for v in flag_triangle]

            pygame.draw.polygon(
                self.canvas,
                color=(255, 0, 0),
                points=self._scale_list_of_tuples(flag_triangle, self.cfg.scale),
            )

            pygame.draw.lines(
                self.canvas,
                color=(0, 0, 0),
                closed=True,
                points=self._scale_list_of_tuples(flag_triangle, self.cfg.scale),
            )

            flag_pole = [trans * v for v in [(flag_x, flag_y), (flag_x, flag_y + flag_pole_length)]]

            pygame.draw.lines(
                self.canvas,
                color=(128, 128, 128),
                closed=True,
                points=self._scale_list_of_tuples(flag_pole, self.cfg.scale),
            )

    def _draw_marker(self, x, y, theta, color: Tuple[int, ...] = (0, 0, 0)):
        """Draw a marker

        Args:
            x (_type_): _description_
            y (_type_): _description_
            theta (_type_): _description_
        """

        offset = 0.2

        cross_vert = [
            (x + offset * np.sin(theta), y - offset * np.cos(theta)),
            (x - offset * np.sin(theta), y + offset * np.cos(theta)),
        ]

        cross_horiz = [
            (x - offset * np.cos(theta), y - offset * np.sin(theta)),
            (x + offset * np.cos(theta), y + offset * np.sin(theta)),
        ]

        pygame.draw.lines(
            self.canvas,
            color=color,
            closed=False,
            points=self._scale_list_of_tuples(cross_horiz, self.cfg.scale),
            width=2,
        )
        pygame.draw.lines(
            self.canvas,
            color=color,
            closed=False,
            points=self._scale_list_of_tuples(cross_vert, self.cfg.scale),
            width=2,
        )

    ## MISC. HELPERS

    @staticmethod
    def _cast_tuple_to_int(t: Tuple) -> Tuple:
        return tuple([int(item) for item in t])

    @staticmethod
    def _scale_list_of_tuples(l: List[Tuple], scale: int) -> List[Tuple]:
        out = []
        for t in l:
            out.append(tuple([item * scale for item in t]))
        return out

    def adjust_dynamics(self, body, **kwargs):
        """Adjust dynamic parameters of a body"""
        # position is weird, only doing velocities and angles
        if kwargs.get("x_dot"):
            body.linearVelocity.x = kwargs["x_dot"]
        if kwargs.get("y_dot"):
            body.linearVelocity.y = kwargs["y_dot"]
        if kwargs.get("theta"):
            body.angle = kwargs["theta"]
        if kwargs.get("theta_dot"):
            body.angularVelocity = kwargs["theta_dot"]

        self.state = self.__generate_state()
