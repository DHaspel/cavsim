#! /opt/conda/bin/python3
""" Pipe class implementing the actual pipe simulation calculations """

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


from typing import Optional
import numpy as np
from .base_pipe import BasePipe
from ..base.connectors.connector import Connector
from ..measure import Measure
from ..base.channels.import_channel import ImportChannel
from ..base.channels.export_channel import ExportChannel


class Pipe(BasePipe):  # pylint: disable=too-many-instance-attributes
    """
    Pipe class implementing the pipe simulation calculations
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
            cavitation_factor: float = None,
            initial_velocity: float = None,
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
        super(Pipe, self).__init__(diameter, length, wall_thickness, bulk_modulus, roughness)
        if inner_points is not None and not isinstance(inner_points, int):
            raise TypeError('Wrong type for parameter inner_points ({} != {})'.format(type(inner_points), int))
        if inner_points is not None and inner_points < 3:
            raise ValueError('Number of inner points ({}) needs to greater than 2!'.format(inner_points))

        # Register special cavitation factors
        self._cavitation_factor = cavitation_factor
        self._initial_velocity = initial_velocity

        # Register internal fields
        self._inner_points = inner_points
        self._pressure: np.ndarray = self.field_create('pressure', 3)
        self._velocity: np.ndarray = self.field_create('velocity', 3)
        self._velocity_u: np.ndarray = self.field_create('velocity_u', 3)
        self.field_create('reynolds', 3)
        self.field_create('reynolds_u', 3)
        self.field_create('brunone', 3)
        self.field_create('brunone_u', 3)
        self.field_create('darcy_friction_factor', 3)
        self.field_create('darcy_friction_factor_u', 3)
        self._friction_steady = self.field_create('friction_steady', 3)
        self._friction_steady_u = self.field_create('friction_steady_u', 3)
        self._friction_unsteady_a = self.field_create('friction_unsteady_a', 3)
        self._friction_unsteady_a_u = self.field_create('friction_unsteady_a_u', 3)
        self._friction_unsteady_b = self.field_create('friction_unsteady_b', 3)
        self._friction_unsteady_b_u = self.field_create('friction_unsteady_b_u', 3)
        self._sos: np.ndarray = self.field_create('speed_of_sound', 3)
        self._initial_pressure = initial_pressure
        self._vapor_volume = self.field_create('vapor', 3)

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
            ExportChannel(Measure.frictionCurrent, lambda: self._friction_steady[0, 1] + self._friction_unsteady_b[0, 1]),
            ExportChannel(Measure.frictionLast, lambda: self._friction_steady[1, 1] + self._friction_unsteady_b[1, 1]),
            ExportChannel(Measure.BPspeedOfSoundCurrent, lambda: self._sos[0, 0]),
            ExportChannel(Measure.BPspeedOfSoundLast, lambda: self._sos[1, 0]),
        ])
        # Create the right connector
        self._right: Connector = Connector(self, [
            ExportChannel(Measure.deltaX, lambda: self._delta_x),
            ImportChannel(Measure.boundaryPoint, False),
            ExportChannel(Measure.diameter, lambda: self.diameter),
            ExportChannel(Measure.length, lambda: self.length),
            ExportChannel(Measure.area, lambda: self.area),
            ImportChannel(Measure.pressureLast, False),
            ExportChannel(Measure.pressureCurrent, lambda: self._pressure[0, -2]),
            ExportChannel(Measure.pressureLast, lambda: self._pressure[1, -2]),
            ExportChannel(Measure.pressureLast2, lambda: self._pressure[2, -2]),
            ImportChannel(Measure.velocityMinusCurrent, False),
            ImportChannel(Measure.velocityMinusLast, False),
            ExportChannel(Measure.velocityPlusCurrent, lambda: self._velocity[0, -2]),
            ExportChannel(Measure.velocityPlusLast, lambda: self._velocity[1, -2]),
            ExportChannel(Measure.frictionCurrent, lambda: self._friction_steady[0, -2] + self._friction_unsteady_a[0, -2]),
            ExportChannel(Measure.frictionLast, lambda: self._friction_steady[1, -2] + self._friction_unsteady_a[1, -2]),
            ExportChannel(Measure.BPspeedOfSoundCurrent, lambda: self._sos[0, -1]),
            ExportChannel(Measure.BPspeedOfSoundLast, lambda: self._sos[1, -1]),
        ])

    @property
    def left(self) -> Connector:
        """
        Left connector property

        :return: Left sided connector of the pipe
        """
        return self._left

    @property
    def right(self) -> Connector:
        """
        Right connector property

        :return: Right sided connector of the pipe
        """
        return self._right

    @property
    def initial_velocity(self) -> float:
        """

        """
        return self._initial_velocity

    @property
    def cavitation_factor(self) -> float:
        """
        Cavitation factor is between 0 and 1, used for two phase calculation only

        :return: The cavitation factor
        """
        return self._cavitation_factor

    @property
    def initial_pressure(self) -> float:
        """
        Condition pressure the normal values are given for

        :return: Pressure of normal conditions [Pa]
        """
        return self._initial_pressure

    @staticmethod
    def b2(self):
        """

        :return: 1 / 4.0
        """
        return 1 / 4.0

    def get_max_delta_t(self) -> Optional[float]:
        """
        Method to return the maximum allowed timestep width for this component

        :return: Maximum allowed timestep width or None if any is suitable
        """
        n_min = self._inner_points if self._inner_points is not None else 3
        result = self.length / ((n_min + 1) * self.norm_speed_of_sound)
        return result

    def discretize(self, delta_t: float) -> None:
        """
        Method handling the discretization of the component (for a given timestep width)

        :param delta_t: Timestep width to discretize for
        :raises ValueError: Timestep too large to fit at least 3 inner points
        """
        self._delta_t = delta_t
        nodes = int(np.ceil(self.length / (self.speed_of_sound(pressure=self.initial_pressure, temperature=None) * delta_t)) - 1)
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
        if self.initial_velocity is not None:
            self.field('velocity')[:, :] = self.initial_velocity * np.ones(self.field('velocity').shape)[:, :]
        else:
            self.field('velocity')[:, :] = np.zeros(self.field('velocity').shape)[:, :]
        self.field('velocity_u')[:, :] = self.field('velocity')[:, :]
        self.field('vapor')[:, :] = np.zeros(self.field('vapor').shape)[:, :]
        if self.initial_velocity is not None:
            self.field('velocity')

        # Initialize derived properties
        for _ in range(2):
            self._calculate_reynolds()
            self._calculate_friction()
            self._calculate_speed_of_sound()
            self._calculate_reynolds_u()
            self._calculate_reynolds()
            self.fields_move()

    def prepare_next_timestep(self, delta_t: float, next_total_time: float) -> None:
        """
        Prepare the internal state for the next timestep to be calculated

        :param delta_t: Timestep width for the next timestep
        :param next_total_time: Total simulation time at the end of the next timestep
        """
        # Shift all internal fields
        self.fields_move()

    def exchange_last_boundaries(self) -> None:
        """
        Exchange the boundary values from the last time steps
        """
        # Exchange previous values with the left boundary
        self._pressure[1, 0] = self.left.value(Measure.pressureLast)
        self._velocity[1, 0] = self.left.value(Measure.velocityPlusLast)
        self._velocity_u[1, 0] = self.left.value(Measure.velocityPlusLast)
        # Exchange previous values with the right boundary
        self._pressure[1, -1] = self.right.value(Measure.pressureLast)
        self._velocity[1, -1] = -self.right.value(Measure.velocityMinusLast)
        self._velocity_u[1, -1] = -self.right.value(Measure.velocityMinusLast)

    def finalize_current_timestep(self) -> None:
        """
        Method to perform final calculations at the end of the current timestep
        """
        # Exchange current values
        self._velocity[0, 0] = self.left.value(Measure.velocityPlusCurrent)
        self._velocity[0, -1] = -self.right.value(Measure.velocityMinusCurrent)
        self._velocity_u[0, 0] = self.left.value(Measure.velocityPlusCurrent)
        self._velocity_u[0, -1] = -self.right.value(Measure.velocityMinusCurrent)
        # Calculate static values
        self._calculate_reynolds()
        self._calculate_reynolds_u()
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
        #self._calculate_velocity()

        # Check calculation case
        # Check is there an index with pressure below vapor pressure or is there an existing vapor bubble
        if np.any(self.field_slice('vapor', 1, 0)[:] > 0.0):
            # loop over all interior pipe nodes
        #    print('cavitation')
            self._calculate_cavitation_velocity()
        else:
            if np.any(self.field_slice('pressure', 0, 0)[:] <= self.fluid.vapor_pressure()):
        #        print('still cavitation')
                self._calculate_cavitation_velocity()
            else:
        #        #print('test')
                self._calculate_velocity()
                #self.field_slice('velocity', 0, 0)[:] = self.field_slice('velocity_u', 0, 0)[:]

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

        # Calculate steady friction
        self._calculate_darcy_friction_factor_u()
        self._calculate_friction_steady_u()
        # Calculate unsteady friction
        self._calculate_brunone_u()
        self._calculate_unsteady_friction_b_u()
        self._calculate_unsteady_friction_a_u()

    def _calculate_pressure(self) -> None:
        """
        Calculate the pressure of the current time step
        """
        # Get the input fields
        pressure_center = self.field_slice('pressure', 1, 0)
        pressure_a = self.field_slice('pressure', 1, -1)
        pressure_b = self.field_slice('pressure', 1, +1)
        velocity_a = self.field_slice('velocity', 1, -1)
        velocity_b = self.field_slice('velocity_u', 1, +1)
        friction_a = self.field_slice('friction_steady', 1, -1) + self.field_slice('friction_unsteady_a', 1, -1)
        friction_b = self.field_slice('friction_steady_u', 1, +1) + self.field_slice('friction_unsteady_b_u', 1, +1)
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
        index = np.where(result <= self.fluid.vapor_pressure())
        # Store/return the calculated result
        self.field_slice('pressure', 0, 0)[:] = result[:]
        result[index] = self.fluid.vapor_pressure()
        self.field_slice('pressure', 0, 0)[index] = result[index]
        index2 = np.where(self.field_slice('vapor', 1, 0) > 0.0)
        result[index2] = self.fluid.vapor_pressure()
        self.field_slice('pressure', 0, 0)[index2] = result[index2]

    def _calculate_velocity(self) -> None:
        """
        Calculate the velocity of the current time step
        """
        # Get the input fields
        pressure_center = self.field_slice('pressure', 1, 0)
        pressure_a = self.field_slice('pressure', 1, -1)
        pressure_b = self.field_slice('pressure', 1, +1)
        velocity_a = self.field_slice('velocity', 1, -1)
        velocity_b = self.field_slice('velocity_u', 1, +1)
        friction_a = self.field_slice('friction_steady', 1, -1) + self.field_slice('friction_unsteady_a', 1, -1)
        friction_b = self.field_slice('friction_steady_u', 1, +1) + self.field_slice('friction_unsteady_b_u', 1, +1)
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
        self.field_slice('velocity_u', 0, 0)[:] = result[:]

    def _calculate_cav_velocity(self, index) -> None:
        """
        Calculate the velocity of the current time step
        """
        # Get the input fields
        pressure_center = self.field_slice('pressure', 1, 0)
        pressure_a = self.field_slice('pressure', 1, -1)
        pressure_b = self.field_slice('pressure', 1, +1)
        velocity_a = self.field_slice('velocity', 1, -1)
        velocity_b = self.field_slice('velocity_u', 1, +1)
        friction_a = self.field_slice('friction_steady', 1, -1) + self.field_slice('friction_unsteady_a', 1, -1)
        friction_b = self.field_slice('friction_steady_u', 1, +1) + self.field_slice('friction_unsteady_b_u', 1, +1)

        # Calculate fluid properties
        speed_of_sound = self.speed_of_sound(pressure=pressure_center[index], temperature=None)
        density = self.fluid.density(pressure=pressure_center[index], temperature=None)

        # Calculate properties at cavitation points
        pressure_center = pressure_center[index]
        pressure_a = pressure_a[index]
        pressure_b = pressure_b[index]
        velocity_a = velocity_a[index]
        velocity_b = velocity_b[index]
        friction_a = friction_a[index]
        friction_b = friction_b[index]

        # Calculate the reynolds number
        result = 0.5 * (
            (velocity_a + velocity_b)
            + ((1.0 / (speed_of_sound * density)) * (pressure_a - pressure_b))
            - (self._delta_t * (friction_a + friction_b))
            # todo: height terms
        )
        # Store/return the calculated result
        self.field_slice('velocity', 0, 0)[index] = result[:]
        self.field_slice('velocity_u', 0, 0)[index] = result[:]

    def _calculate_cavitation_pressure(self, index) -> None:
        """
        Calculate the pressure of the current time step
        """
        # Get the input fields
        pressure_center = self.field_slice('pressure', 1, 0)
        pressure_a = self.field_slice('pressure', 1, -1)
        pressure_b = self.field_slice('pressure', 1, +1)
        velocity_a = self.field_slice('velocity', 1, -1)
        velocity_b = self.field_slice('velocity_u', 1, +1)
        friction_a = self.field_slice('friction_steady', 1, -1) + self.field_slice('friction_unsteady_a', 1, -1)
        friction_b = self.field_slice('friction_steady_u', 1, +1) + self.field_slice('friction_unsteady_b_u', 1, +1)

        # Calculate fluid properties
        speed_of_sound = self.speed_of_sound(pressure=pressure_center[index], temperature=None)
        density = self.fluid.density(pressure=pressure_center[index], temperature=None)

        # Calculate properties at cavitation points
        pressure_center = pressure_center[index]
        pressure_a = pressure_a[index]
        pressure_b = pressure_b[index]
        velocity_a = velocity_a[index]
        velocity_b = velocity_b[index]
        friction_a = friction_a[index]
        friction_b = friction_b[index]

        # Calculate the reynolds number
        result = 0.5 * (
                (speed_of_sound * density * (velocity_a - velocity_b))
                + (pressure_a + pressure_b)
                + (self._delta_x * density * (friction_b - friction_a))
            # todo: height terms
        )
        # Store/return the calculated result
        self.field_slice('pressure', 0, 0)[index] = result[:]

    def _calculate_cavitation_velocity(self) -> None:
        """
        Calculate the velocity of the current time step
        """
        # Mark the positions with cavitation
        index = np.argwhere(self.field_slice('pressure', 0, 0)[:] <= self.fluid.vapor_pressure())

        # Get the input fields
        self.field_slice('pressure', 0, 0)[index] = self.fluid.vapor_pressure()
        pressure_center = self.field_slice('pressure', 1, 0)
        pressure_a = self.field_slice('pressure', 1, -1)
        pressure_b = self.field_slice('pressure', 1, +1)
        velocity_a = self.field_slice('velocity', 1, -1)
        velocity_b = self.field_slice('velocity_u', 1, +1)
        friction_a = self.field_slice('friction_steady', 1, -1) + self.field_slice('friction_unsteady_a', 1, -1)
        friction_b = self.field_slice('friction_steady_u', 1, +1) + self.field_slice('friction_unsteady_b_u', 1, +1)

        # Calculate fluid properties
        speed_of_sound = self.speed_of_sound(pressure=pressure_center, temperature=None)
        density = self.fluid.density(pressure=pressure_center, temperature=None)

        # Calculate the velocity in the non cavitating points
        result = 0.5 * (
            (velocity_a + velocity_b)
            + ((1.0 / (speed_of_sound * density)) * (pressure_a - pressure_b))
            - (self._delta_t * (friction_a + friction_b))
            # todo: height terms
        )
        self.field_slice('velocity', 0, 0)[:] = result[:]
        self.field_slice('velocity_u', 0, 0)[:] = result[:]

        # Calculate the left sided solutions
        left_result = (1.0 / (speed_of_sound[index] * density[index]) * (pressure_a[index] - self.fluid.vapor_pressure())
                       + velocity_a[index]
                       - friction_a[index] * self._delta_t
                       )

        # Store/return the calculated left_result
        #print(left_result.shape)
        #print(self.field_slice('velocity_u').shape)
        self.field_slice('velocity_u', 0, 0)[index] = left_result[:]

        # Calculate the right sided solutions

        right_result = (1.0 / (speed_of_sound[index] * density[index]) * (self.fluid.vapor_pressure() - pressure_b[index])
                        + velocity_b[index]
                        - friction_b[index] * self._delta_t
                        )
        # Store/return the calculated right_result
        self.field_slice('velocity', 0, 0)[index] = right_result[:]

        # Get former velocities
        former_right_velocity = self.field_slice('velocity', 1, 0)[index]
        former_left_velocity = self.field_slice('velocity_u', 1, 0)[index]

        # Calculate size of the vapor bubble
        try:
            vapor_result = (self.field_slice('vapor', 1, 0)[index]
                            + 2.0 * self._delta_t * (self.cavitation_factor * (right_result - left_result)
                                                     + (1 - self.cavitation_factor)
                                                     * (former_right_velocity - former_left_velocity)) * self.area
                            )
        except:
            print("Fatal Error")
            print("Cavitation_factor is {}").format(self.cavitation_factor)

        # Store/return the calculated vapor volumes
        self.field_slice('vapor', 0, 0)[index] = vapor_result
        # Are there any positions where the vapor bubbles disappear?
        new_index = np.argwhere(self.field_slice('vapor', 0, 0)[:] < 0.0)
        # Vapor volume is now zero in the areas without any cavitation
        self.field_slice('vapor', 0, 0)[new_index] = 0.0
        # Perform normal calculation for the pressure and velocity
        self._calculate_cavitation_pressure(new_index)
        self._calculate_cav_velocity(new_index)

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

    def _calculate_reynolds_u(self) -> None:
        """
        Calculate the Reynolds number based on the values from the previous time step
        """
        # Get the input fields
        pressure = self.field_wide_slice('pressure', 0)
        velocity = self.field_wide_slice('velocity_u', 0)
        # Calculate fluid properties
        viscosity = self.fluid.viscosity(temperature=None, shear_rate=None)
        density = self.fluid.density(pressure=pressure, temperature=None)
        # Calculate the reynolds number
        result = (density * np.abs(velocity) * self.diameter) / viscosity
        # Store/return the calculated result
        self.field_wide_slice('reynolds_u', 0)[:] = result[:]

    def _calculate_brunone_u(self) -> None:
        """
        Calculate the Brunone factor for unsteady friction
        """
        # Get the input fields
        reynolds = self.field_wide_slice('reynolds_u', 0)
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
        self.field_wide_slice('brunone_u', 0)[:] = result[:]

    def _calculate_unsteady_friction_b_u(self) -> None:
        """
        Calculate the unsteady friction to right side
        """
        # Get the input fields
        brunone = self.field_ext_slice('brunone_u', 0, 1)
        velocity_b = self.field_ext_slice('velocity_u', 0, 1)
        velocity_bb = self.field_ext_slice('velocity_u', 1, 1)
        velocity_p = self.field_ext_slice('velocity_u', 0, 0)
        pressure_b = self.field_ext_slice('pressure', 0, 1)
        # Calculate fluid properties
        speed_of_sound = self.speed_of_sound(pressure=pressure_b, temperature=None)
        # Calculate the friction
        vdt = (velocity_b - velocity_bb) / self._delta_t
        vdx = (velocity_b - velocity_p) / self._delta_x
        result = brunone * (vdt + (speed_of_sound * np.sign(velocity_b * vdx) * vdx))
        # Store/return the calculated result
        self.field_ext_slice('friction_unsteady_b_u', 0, 1)[:] = result[:]

    def _calculate_unsteady_friction_a_u(self) -> None:
        """
        Calculate the unsteady friction to left side
        """
        # Get the input fields
        brunone = self.field_ext_slice('brunone', 0, 0)
        velocity_a = self.field_ext_slice('velocity', 0, 0)
        velocity_aa = self.field_ext_slice('velocity', 1, 0)
        velocity_p = self.field_ext_slice('velocity_u', 0, 1)
        pressure_a = self.field_ext_slice('pressure', 0, 0)
        # Calculate fluid properties
        speed_of_sound = self.speed_of_sound(pressure=pressure_a, temperature=None)
        # Calculate the friction
        vdt = (velocity_a - velocity_aa) / self._delta_t
        vdx = (velocity_p - velocity_a) / self._delta_x
        result = brunone * (vdt + (speed_of_sound * np.sign(velocity_a * vdx) * vdx))
        # Store/return the calculated result
        self.field_ext_slice('friction_unsteady_a_u', 0, 0)[:] = result[:]

    def _calculate_darcy_friction_factor_u(self) -> None:
        """
        Calculates darcy's friction coefficient within the pipe
        """
        # Get the input fields
        reynolds = self.field_wide_slice('reynolds_u', 0)
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
        self.field_wide_slice('darcy_friction_factor_u', 0)[:] = result[:]

    def _calculate_friction_steady_u(self) -> None:
        """
        Calculate the steady friction using darcy's factor
        """
        # Get the input fields
        velocity = self.field_wide_slice('velocity_u', 0)
        friction_factor = self.field_wide_slice('darcy_friction_factor_u', 0)
        # Calculate the friction
        result = (friction_factor / (2.0 * self.diameter)) * np.abs(velocity) * velocity
        # Store/return the calculated result
        self.field_wide_slice('friction_steady_u', 0)[:] = result[:]
