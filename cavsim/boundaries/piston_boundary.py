#! /opt/conda/bin/python3
""" Pipe class implementing a right boundary with given velocity """

# Copyright 2019 FAU-iPAT (http://ipat.uni-erlangen.de/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Optional, Union
import numpy as np
from cavsim.boundaries.base_boundary import BaseBoundary, BoundaryFunction
from cavsim.base.connectors.connector import Connector
from cavsim.measure import Measure
from cavsim.base.channels.import_channel import ImportChannel
from cavsim.base.channels.export_channel import ExportChannel
from cavsim.pipes.base_pipe import BasePipe


class PumpBoundary(BasePipe, BaseBoundary):
    """
    Pump boundary with a moving wall and interpolation of time and Space on the grid to performe exact calculations
    """

    def __init__(
            self,
            diameter: float,
            length: float,
            wall_thickness: float,
            bulk_modulus: float,
            roughness: float,
            inner_points: int = None,
            initial_pressure: float = None,
            rpm: float = None,
            radius: float = None,
            rod_ratio: float = None,
            phase_angle: float = None,

    ) -> None:
        """
        Initialization of the class

        :param diameter: Diameter of the fluid volume [m]
        :param length: Length of the pipe [m]
        :param wall_thickness: Thickness of the wall [m]
        :param bulk_modulus: Bulk modulus of wall material [Pa]
        :param roughness: Roughness of the wall [m]
        :param inner_points: Minimal number of inner points for discretization
        :raises TypeError: Wrong type of at least one parameter
        :raises ValueError: Value of at least one parameter out of bounds
        """
        super(PumpBoundary, self).__init__(diameter, length, wall_thickness, bulk_modulus, roughness)
        if inner_points is not None and not isinstance(inner_points, int):
            raise TypeError('Wrong type for parameter inner_points ({} != {}'.format(type(inner_points), int))
        if inner_points is not None and inner_points < 3:
            raise ValueError('Number of inner points ({}) needs to greater than 2!'.format(inner_points))
        # Register internal fields
        self._inner_points = inner_points
        self._pressure: np.ndarray = self.field_create('pressure', 3)
        self._velocity: np.ndarray = self.field_create('velocity', 3)
        self.field_create('reynolds', 3)
        self.field_create('brunone', 3)
        self.field_create('darcy_friction_factor', 3)
        self._friction_steady = self.field_create('friction_steady', 3)
        self._friction_unsteady_a = self.field_create('friction_unsteady_a', 3)
        self._friction_unsteady_b = self.field_create('friction_unsteady_b', 3)
        self._sos: np.ndarray = self.field_create('speed_of_sound', 3)
        self.piston_position: np.ndarray = self.field_create('piston_position', 3)
        self.piston_velocity: np.ndarray = self.field_create('piston_velocity', 3)
        self.x_x: np.ndarray = self.field_create('x_x', 3)
        self.x_a: np.ndarray = self.field_create('x_a', 3)
        self.x_b: np.ndarray = self.field_create('x_b', 3)
        self.left_index: np.ndarray = self.field_create('left_index', 3)
        self.right_index: np.ndarray = self.field_create('right_index', 3)
        self._initial_pressure = initial_pressure
        self._number_of_points = inner_points + 2
        self._number_of_former_points = inner_points + 2
        self._chamber_length = 0.0
        self._former_chamber_length = 0.0
        self._rpm = rpm
        self._radius = radius
        self._rod_ratio = rod_ratio
        self._phase_angle = phase_angle
        self.time = 0.0
        self._dx = 0.0
        self._former_dx = 0.0

        # Create the left connector
        self._left: Connector = Connector(self, [
            ExportChannel(Measure.deltaX, lambda: self._delta_x),
            ImportChannel(Measure.boundaryPoint, False),
            ExportChannel(Measure.diameter, lambda: self.diameter),
            ExportChannel(Measure.length, lambda: self.length),
            ExportChannel(Measure.area, lambda: self.area),
            ImportChannel(Measure.pressureLast, False),
            ExportChannel(Measure.pressureCurrent, lambda: self._pressure[0, 1]),
            ExportChannel(Measure.pressureLast, lambda: self._pressure[1, 1]),
            ExportChannel(Measure.pressureLast2, lambda: self._pressure[2, 1]),
            ImportChannel(Measure.velocityPlusCurrent, False),
            ImportChannel(Measure.velocityPlusLast, False),
            ExportChannel(Measure.velocityMinusCurrent, lambda: -self._velocity[0, 1]),
            ExportChannel(Measure.velocityMinusLast, lambda: -self._velocity[1, 1]),
            ExportChannel(Measure.frictionCurrent,
                          lambda: self._friction_steady[0, 1] + self._friction_unsteady_b[0, 1]),
            ExportChannel(Measure.frictionLast, lambda: self._friction_steady[1, 1] + self._friction_unsteady_b[1, 1]),
            ExportChannel(Measure.BPspeedOfSoundCurrent, lambda: self._sos[0, 0]),
            ExportChannel(Measure.BPspeedOfSoundLast, lambda: self._sos[1, 0]),
        ])

    @property
    def left(self) -> Connector:
        """
        Left connector property

        :return: Left sided connector of the pipe
        """
        return self._left

    @property
    def number_of_points(self) -> int:

        return self._number_of_points

    @property
    def number_of_former_points(self) -> int:

        return self._number_of_former_points

    @property
    def minimum_length(self) -> float:

        return self.length - 2.0 * self.radius

    @property
    def angular_velocity(self) -> float:
        """
        angular velocity of the piston

        :return: angular velocity
        """
        return (self.rpm / 60) * 2.0 * np.pi

    @property
    def phase_angle(self) -> float:
        """
        Phase angle of the pump kinematics

        :return: Phase angle [°]
        """
        return self._phase_angle

    @property
    def rod_ratio(self) -> float:
        """
        Rod ratio of the pump kinematics

        :return: Rod ratio [-]
        """
        return self._rod_ratio

    @property
    def radius(self):
        """
        Radius of the piston motion

        :return: Radius [m]
        """
        return self._radius

    @property
    def rpm(self) -> float:
        """
        Rounds per Minute of the pump

        :return: Rounds per Minute [1/min]

        """
        return self._rpm

    @property
    def initial_pressure(self) -> float:
        """
        Condition pressure the normal values are given for

        :return: Pressure of normal conditions [Pa]
        """
        return self._initial_pressure

    def get_max_delta_t(self) -> Optional[float]:
        """
        Method to return the maximum allowed timestep width for this component

        :return: Maximum allowed timestep width or None if any is suitable
        """
        n_min = self._inner_points if self._inner_points is not None else 3
        result = self.length / ((n_min + 1) * self.fluid.norm_speed_of_sound)
        return result

    def discretize(self, delta_t: float) -> None:
        """
        Method handling the discretization of the component (for a given timestep width)

        :param delta_t: Timestep width to discretize for
        :raises ValueError: Timestep too large to fit at least 3 inner points
        """
        self._delta_t = delta_t
        nodes = int(np.ceil(self.length / (self.fluid.norm_speed_of_sound * delta_t)) - 1)
        if nodes < 3:
            raise ValueError('Timestep to large!')
        self._delta_x = self.length / float(nodes + 1)
        self.fields_resize(nodes + 2)

    def initialize(self) -> None:
        """
        Initialize the internal state of the component (after discretization was called)
        """
        if self.initial_pressure is not None:
            self.field('pressure')[:, :] = self.initial_pressure * np.ones(self.field('pressure').shape)[:, :]
        else:
            self.field('pressure')[:, :] = self.fluid.initial_pressure * np.ones(self.field('pressure').shape)[:, :]

        self.field('velocity')[:, :] = np.zeros(self.field('velocity').shape)[:, :]
        self._former_dx = self._delta_x
        self._dx = self._delta_x

        # Initialize derived properties
        for _ in range(2):
            self._calculate_reynolds()
            self._calculate_friction()
            self._calculate_speed_of_sound()
            self._calculate_current_piston_velocity()
            self._calculate_current_piston_coordinate()
            self._calculate_number_of_points()
            self._calculate_new_mesh()
            self.fields_move()

    def _calculate_current_piston_coordinate(self) -> None:

        self.piston_position[0, 0] = (
                self.radius * (1.0
                               - np.cos(self.angular_velocity * self.time + self.phase_angle)
                               + self.rod_ratio / 2.0 * np.sin(self.angular_velocity * self.time + self.phase_angle)**2)
                + self.minimum_length
        )

    def _calculate_current_piston_velocity(self) -> None:

        self.piston_velocity[0, 0] = self.radius * self.angular_velocity * (
            np.sin(self.angular_velocity * self.time + self.phase_angle)
            + self.rod_ratio / 2.0 * np.sin(2.0 * (self.angular_velocity * self.time + self.phase_angle))
        )

    def _calculate_number_of_points(self):

        self._number_of_points = max(np.floor(self.piston_position[0, 0] / self._delta_x - 2), 3).astype(int)
        print(self._number_of_points)
        self.delta_x_current = self.piston_position[0, 0] / (self._number_of_points + 2)

    def _calculate_new_mesh(self):
        # Create a new spatial grid as linspace from the piston position towards the piston surface
        # Create the necessary fields
        self.field_slice('x_x', 0, 0)[:self._number_of_points] = (
            np.linspace(self.piston_position[0, 0], 0.0, self._number_of_points)
                                            )
        x_x = self.field_slice('x_x', 0, 0)

        self._dx = x_x[1] - x_x[0]
        dt = self._delta_t
        sos = self._delta_x / self._delta_t
        velocity = self.piston_velocity[1, 0]
        # Calculate the left and right sided spatial grid needed to interpolate towards x_x
        self.field_slice('x_a', 1, -1)[:self._number_of_points] = x_x + velocity * dt - (velocity + sos) * dt
        self.field_slice('x_b', 1, +1)[:self._number_of_points] = x_x + velocity * dt - (velocity - sos) * dt
        x_a = self.field_slice('x_a', 1, -1)[:self._number_of_points]
        x_b = self.field_slice('x_b', 1, +1)[:self._number_of_points]

        # Calculate the right indices of the spatial position
        self.field_slice('left_index', 1, -1)[:self._number_of_points] = (
            np.floor((self.piston_position[0, 0] - x_a)
                     / self._former_dx)
        ).astype(int)
        print(self.field_slice('left_index', 1, -1)[:self._number_of_points])
        self.field_slice('right_index', 1, -1)[:self._number_of_points] = (
            np.floor((self.piston_position[0, 0] - x_b)
                     / self._former_dx)
        ).astype(int)

        # Write the fields into the corresponding indices with mapping
        self.field_slice('delta_x_a', 1, -1)[:self._number_of_points] = (
            self.field_slice('x_x', 1, 0)[:self._number_of_points]
            - self.field_slice('x_a', 1, -1)[self.field_slice('left_index', 1, -1)[:self._number_of_points].astype(int)]
        )
        #self.field_slice('delta_x_b', 1, +1)[:self._number_of_points] = (
        #    self.field_slice('x_x', 1, 0)[:self._number_of_points]
        #)

    def _calculate_space_interpolation(self) -> None:

        velocity = self.field_slice('velocity', 1, 0)
        pressure = self.field_slice('pressure', 1, 0)
        velocity_a = self.field_slice('velocity', 1, -1)
        velocity_b = self.field_slice('velocity', 1, +1)
        pressure_a = self.field_slice('pressure', 1, -1)
        pressure_b = self.field_slice('pressure', 1, +1)
        speed_of_sound_a = self._delta_x / self._delta_t
        speed_of_sound_b = self._delta_x / self._delta_t

        #self.field_slice('delta_x_a', 1, -1)[:] = (
        #        ((velocity + speed_of_sound_a)
        #         / (self._delta_x / self._delta_t
        #            + 1.0 / 2.0 * (velocity - velocity_a)))
        #        * self._delta_x
        #)
        #self.field_slice('delta_x_b', 1, +1)[:] = (
        #        ((velocity - speed_of_sound_b)
        #         / (self._delta_x / self._delta_t
        #            + 1.0 / 2.0 * (velocity_b - velocity)))
        #        * self._delta_x
        #)

        delta_x_a = self.field_slice('delta_x_a', 1, -1)[:]
        delta_x_b = self.field_slice('delta_x_b', 1, +1)[:]
        #self.field_slice('velocity')

        self.field_slice('pressure_a', 1, -1)[:] = (pressure
                                                    - ((pressure - pressure_a) / self._delta_x)
                                                    * delta_x_a)
        self.field_slice('pressure_b', 1, +1)[:] = (pressure
                                                    - ((pressure_b - pressure) / self._delta_x)
                                                    * delta_x_b)
        self.field_slice('velocity_a', 1, -1)[:] = (velocity
                                                    - ((velocity - velocity_a) / self._delta_x)
                                                    * delta_x_a)
        self.field_slice('velocity_b', 1, +1)[:] = (velocity
                                                    - ((velocity_b - velocity) / self._delta_x)
                                                    * delta_x_b)

    def prepare_next_timestep(self, delta_t: float, next_total_time: float) -> None:
        """
        Prepare the internal state for the next timestep to be calculated

        :param delta_t: Timestep width for the next timestep
        :param next_total_time: Total simulation time at the end of the next timestep
        """
        # Shift all internal fields
        self.fields_move()
        self.time = next_total_time
        #self._velocity[0, 1] = self._boundary(next_total_time) if callable(self._boundary) else self._boundary
        self._former_dx = self._dx
        self._calculate_current_piston_velocity()
        self._calculate_current_piston_coordinate()
        self._calculate_number_of_points()
        self._calculate_new_mesh()
        self.field_ext_slice('velocity', 0, 0)[self._number_of_points + 1] = (
            self._boundary(next_total_time) if callable(self._boundary) else self._boundary
        )

    def exchange_last_boundaries(self) -> None:
        """
        Exchange the boundary values from the last time steps
        """
        # Exchange previous values with the left boundary
        # Only one side is needed in this case because the bc is given
        self._pressure[1, 0] = self.left.value(Measure.pressureLast)
        self._velocity[1, 0] = self.left.value(Measure.velocityPlusLast)
        # Exchange previous values with the right boundary
        #self._pressure[1, -1] = self.right.value(Measure.pressureLast)
        #self._velocity[1, -1] = -self.right.value(Measure.velocityMinusLast)

    def finalize_current_timestep(self) -> None:
        """
        Method to perform final calculations at the end of the current timestep
        """
        # Exchange current values
        self._velocity[0, 0] = self.left.value(Measure.velocityPlusCurrent)
        # Calculate static values
        self._calculate_reynolds()
        self._calculate_friction()
        self._calculate_speed_of_sound()

    def prepare_next_inner_iteration(self, iteration: int) -> None:
        """
        Method to prepare the internal state for the next inner iteration of the current timestep

        :param iteration: Number of the next inner iteration to prepare for
        """

    def exchange_current_boundaries(self) -> None:
        """
        Exchange boundary values from the current time step
        """

    def calculate_next_inner_iteration(self, iteration: int) -> bool:
        """
        Method to do the calculations of the next inner iteration

        :param iteration: Number of the next inner iteration
        :return: Whether this component needs another inner iteration afterwards
        """

        self._calculate_pressure()
        self._calculate_velocity()

        return False

    def _calculate_speed_of_sound(self) -> None:
        """
        Calculate the current speed of sound
        """
        pressure = self.field_wide_slice('pressure', 0)
        result = self.speed_of_sound(pressure=pressure, temperature=None)
        self.field_wide_slice('speed_of_sound', 0)[:] = result[:]

    def _calculate_reynolds(self) -> None:
        """
        Calculate the Reynolds number based on the values from the previous time step
        """
        # Get the input fields
        pressure = self.field_wide_slice('pressure', 0)
        velocity = self.field_wide_slice('velocity', 0)
        # Calculate fluid properties
        viscosity = self.fluid.viscosity(temperature=None, shear_rate=None)
        density = self.fluid.density(pressure=pressure, temperature=None)
        # Calculate the reynolds number
        result = (density * np.abs(velocity) * self.diameter) / viscosity
        # Store/return the calculated result
        self.field_wide_slice('reynolds', 0)[:] = result[:]

    def _calculate_darcy_friction_factor(self) -> None:
        """
        Calculates darcy's friction coefficient within the pipe
        """
        # Get the input fields
        reynolds = self.field_wide_slice('reynolds', 0)
        result = np.ones(reynolds.shape)
        # Calculate the friction factor (low Re)
        selector = np.logical_and(reynolds > 0.0, reynolds < 2100.0)
        if np.sum(selector) > 0:
            local_reynolds = reynolds[selector]
            factor = 64.0 / local_reynolds
            result[selector] = factor
        # Calculate the friction factor (high Re)
        selector = (reynolds >= 2100.0)
        if np.sum(selector) > 0:
            local_reynolds = reynolds[selector]
            factor = 10.0 * np.ones(local_reynolds.shape)
            error = np.ones(local_reynolds.shape)
            while np.any(error > 1e-12):
                term1 = self.roughness / (3.7 * self.diameter)
                term2 = 2.51 / (local_reynolds * np.sqrt(factor))
                temp = -2.0 * np.log10(term1 + term2)
                old_factor, factor = factor, np.square(1.0 / temp)
                error = np.abs(factor - old_factor)
            result[selector] = factor
        # Store/return the calculated result
        self.field_wide_slice('darcy_friction_factor', 0)[:] = result[:]

    def _calculate_friction_steady(self) -> None:
        """
        Calculate the steady friction using darcy's factor
        """
        # Get the input fields
        velocity = self.field_wide_slice('velocity', 0)
        friction_factor = self.field_wide_slice('darcy_friction_factor', 0)
        # Calculate the friction
        result = (friction_factor / (2.0 * self.diameter)) * np.abs(velocity) * velocity
        # Store/return the calculated result
        self.field_wide_slice('friction_steady', 0)[:] = result[:]

    def _calculate_friction(self) -> None:
        """
        Calculate the total friction (steady + unsteady)
        """
        # Calculate steady friction
        self._calculate_darcy_friction_factor()
        self._calculate_friction_steady()
        # Calculate unsteady friction
        self._calculate_brunone()
        self._calculate_unsteady_friction_a()
        self._calculate_unsteady_friction_b()

    def _calculate_pressure(self) -> None:
        """
        Calculate the pressure of the current time step
        """
        # Get the input fields
        pressure_center = self.field_slice('pressure', 1, 0)
        pressure_a = self.field_slice('pressure', 1, -1)
        pressure_b = self.field_slice('pressure', 1, +1)
        velocity_a = self.field_slice('velocity', 1, -1)
        velocity_b = self.field_slice('velocity', 1, +1)
        friction_a = self.field_slice('friction_steady', 1, -1) + self.field_slice('friction_unsteady_a', 1, -1)
        friction_b = self.field_slice('friction_steady', 1, +1) + self.field_slice('friction_unsteady_b', 1, +1)
        # Calculate fluid properties
        speed_of_sound = self.speed_of_sound(pressure=pressure_center, temperature=None)
        density = self.fluid.density(pressure=pressure_center, temperature=None)
        # Calculate the reynolds number
        result = 0.5 * (
                (speed_of_sound * density * (velocity_a - velocity_b))
                + (pressure_a + pressure_b)
                + (self._delta_x * density * (friction_b - friction_a))
            # todo: height terms
        )
        # Store/return the calculated result
        self.field_slice('pressure', 0, 0)[:] = result[:]

    def _calculate_velocity(self) -> None:
        """
        Calculate the velocity of the current time step
        """
        # Get the input fields
        pressure_center = self.field_slice('pressure', 1, 0)
        pressure_a = self.field_slice('pressure', 1, -1)
        pressure_b = self.field_slice('pressure', 1, +1)
        velocity_a = self.field_slice('velocity', 1, -1)
        velocity_b = self.field_slice('velocity', 1, +1)
        friction_a = self.field_slice('friction_steady', 1, -1) + self.field_slice('friction_unsteady_a', 1, -1)
        friction_b = self.field_slice('friction_steady', 1, +1) + self.field_slice('friction_unsteady_b', 1, +1)
        # Calculate fluid properties
        speed_of_sound = self.speed_of_sound(pressure=pressure_center, temperature=None)
        density = self.fluid.density(pressure=pressure_center, temperature=None)
        # Calculate the reynolds number
        result = 0.5 * (
                (velocity_a + velocity_b)
                + ((1.0 / (speed_of_sound * density)) * (pressure_a - pressure_b))
                - (self._delta_t * (friction_a + friction_b))
            # todo: height terms
        )
        # Store/return the calculated result
        self.field_slice('velocity', 0, 0)[:] = result[:]

    def _calculate_brunone(self) -> None:
        """
        Calculate the Brunone factor for unsteady friction
        """
        # Get the input fields
        reynolds = self.field_wide_slice('reynolds', 0)
        # Calculate the Brunone factor
        result = 0.000476 * np.ones(reynolds.shape)
        selector = (reynolds >= 2320.0)
        if np.sum(selector) > 0:
            local_reynolds = reynolds[selector]
            factor = 14.3 / np.power(local_reynolds, 0.05)
            factor = 7.41 / np.power(local_reynolds, np.log10(factor))
            result[selector] = factor
        result = np.sqrt(result) / 2.0
        # Store/return the calculated result
        self.field_wide_slice('brunone', 0)[:] = result[:]

    def _calculate_unsteady_friction_a(self) -> None:
        """
        Calculate the unsteady friction to left side
        """
        # Get the input fields
        brunone = self.field_ext_slice('brunone', 0, 0)
        velocity_a = self.field_ext_slice('velocity', 0, 0)
        velocity_aa = self.field_ext_slice('velocity', 1, 0)
        velocity_p = self.field_ext_slice('velocity', 0, 1)
        pressure_a = self.field_ext_slice('pressure', 0, 0)
        # Calculate fluid properties
        speed_of_sound = self.speed_of_sound(pressure=pressure_a, temperature=None)
        # Calculate the friction
        vdt = (velocity_a - velocity_aa) / self._delta_t
        vdx = (velocity_p - velocity_a) / self._delta_x
        result = brunone * (vdt + (speed_of_sound * np.sign(velocity_a * vdx) * vdx))
        # Store/return the calculated result
        self.field_ext_slice('friction_unsteady_a', 0, 0)[:] = result[:]

    def _calculate_unsteady_friction_b(self) -> None:
        """
        Calculate the unsteady friction to right side
        """
        # Get the input fields
        brunone = self.field_ext_slice('brunone', 0, 1)
        velocity_b = self.field_ext_slice('velocity', 0, 1)
        velocity_bb = self.field_ext_slice('velocity', 1, 1)
        velocity_p = self.field_ext_slice('velocity', 0, 0)
        pressure_b = self.field_ext_slice('pressure', 0, 1)
        # Calculate fluid properties
        speed_of_sound = self.speed_of_sound(pressure=pressure_b, temperature=None)
        # Calculate the friction
        vdt = (velocity_b - velocity_bb) / self._delta_t
        vdx = (velocity_b - velocity_p) / self._delta_x
        result = brunone * (vdt + (speed_of_sound * np.sign(velocity_b * vdx) * vdx))
        # Store/return the calculated result
        self.field_ext_slice('friction_unsteady_b', 0, 1)[:] = result[:]
