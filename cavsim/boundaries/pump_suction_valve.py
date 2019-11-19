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
from .base_boundary import BaseBoundary, BoundaryFunction
from ..base.connectors.connector import Connector
from ..measure import Measure
from ..base.channels.import_channel import ImportChannel
from ..base.channels.export_channel import ExportChannel


class ZetaJoint(BaseBoundary):
    """
    Pipe class implementing the pipe simulation calculations
    """

    def __init__(
            self,
            zeta: float,                # -
            density: float,             # kg/m³
            prestressed_force: float,   # N
            spring_stiffness: float,     # N/mm


    ) -> None:
        """
        Initialization of the Class

        :param zeta: zeta value to calculate the flow resistance
        """

        # Register zeta value
        self._zeta =zeta

        super(ZetaJoint, self).__init__()

        # Register internal fields
        self._pressure: np.ndarray = self.field_create('pressure', 3)
        self._velocity: np.ndarray = self.field_create('velocity', 3)
        self._volume_flow: np.ndarray = self.field_create('volume_flow', 3)
        self._sos: np.ndarray = self.field_create('speed_of_sound', 3)
        self._friction = self.field_create('friction', 3)
        self._displacement = self.field_create('displacement', 1)
        self._valve_velocity = self.field_create('valve_velocity', 1)
        self._acceleration = self.field_create('acceleration', 1)
        self._area = np.empty(2)
        self._gravity_force = 0.0
        self._g = 9.81
        self._mat_density = density
        self._prestressed_force = prestressed_force
        self._spring_stiffness = spring_stiffness

        # Create the left connector
        self._left: Connector = Connector(self, [
            ImportChannel(Measure.deltaX, False),
            ExportChannel(Measure.boundaryPoint, lambda: True),
            ImportChannel(Measure.pressureLast, False),
            ImportChannel(Measure.pressureLast2, False),
            ExportChannel(Measure.pressureCurrent, lambda: self._pressure[0, 1]),
            ExportChannel(Measure.pressureLast, lambda: self._pressure[1, 1]),
            ImportChannel(Measure.velocityPlusLast, False),
            ExportChannel(Measure.velocityMinusCurrent, lambda: -self._volume_flow[0, 1]/(self._area[0])),
            ExportChannel(Measure.velocityMinusLast, lambda: -self._volume_flow[1, 1]/(self._area[0])),
            ImportChannel(Measure.frictionLast, False),
            ImportChannel(Measure.BPspeedOfSoundLast, False),
            ImportChannel(Measure.area)
        ])
        # Create the right connector
        self._right: Connector = Connector(self, [
            ImportChannel(Measure.deltaX, False),
            ExportChannel(Measure.boundaryPoint, lambda: True),
            ImportChannel(Measure.pressureLast, False),
            ImportChannel(Measure.pressureLast2, False),
            ExportChannel(Measure.pressureCurrent, lambda: self._pressure[0, -2]),
            ExportChannel(Measure.pressureLast, lambda: self._pressure[1, -2]),
            ImportChannel(Measure.velocityMinusLast, False),
            ExportChannel(Measure.velocityPlusCurrent, lambda: self._volume_flow[0, -2]/(self._area[1])),
            ExportChannel(Measure.velocityPlusLast, lambda: self._volume_flow[0, -2]/(self._area[1])),
            ImportChannel(Measure.frictionLast, False),
            ImportChannel(Measure.BPspeedOfSoundLast, False),
            ImportChannel(Measure.area)
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
    def zeta(self) -> float:
        """
        Zeta value of the resistance

        :return: Zeta value of the resistance
        """
        return self._zeta

    @property
    def density(self):
        """
        Averaged density of the valve property

        :return: Averaged density of the valve
        """
        return self._mat_density

    def spring_stiffness(self):
        """
        Spring stiffness property

        :return: stiffness of the spring
        """
        return self._spring_stiffness

    def spring_force(self):

        return self._spring_force

    def get_max_delta_t(self) -> Optional[float]:
        """
        Method to return the maximum allowed timestep width for this component

        :return: Maximum allowed timestep width or None if any is suitable
        """

    def discretize(self, delta_t: float) -> None:
        """
        Method handling the discretization of the component (for a given timestep width)

        :param delta_t: Timestep width to discretize for
        :raises ValueError: Timestep too large to fit at least 3 inner points
        """
        self._delta_t = delta_t
        self.fields_resize(3)

    def initialize(self) -> None:
        """
        Initialize the internal state of the component (after discretization was called)
        """
        self.field('velocity')[:, :] = np.zeros(self.field('velocity').shape)[:, :]
        self.field('pressure')[:, :] = self.fluid.initial_pressure * np.ones(self.field('pressure').shape)[:, :]
        self.field('friction')[:, :] = np.zeros(self.field('friction').shape)[:, :]
        self.field('speed_of_sound')[:, :] = np.zeros(self.field('friction').shape)[:, :]
        self.field('displacement')[:, :] = np.zeros(self.field('displacement').shape)[:, :]
        self.field('valve_velocity')[:, :] = np.zeros(self.field('valve_velocity').shape)[:, :]
        self.field('acceleration')[:, :] = np.zeros(self.field('acceleration').shape)[:, :]


    def prepare_next_timestep(self, delta_t: float, next_total_time: float) -> None:
        """
        Prepare the internal state for the next timestep to be calculated

        :param delta_t: Timestep width for the next timestep
        :param next_total_time: Total simulation time at the end of the next timestep
        """
        # Shift all internal fields
        self.fields_move()
        # Set fixed pressure value

    def exchange_last_boundaries(self) -> None:
        """
        Exchange boundary values from previous time steps
        """
        # Exchange previous values with the left boundary
        self._pressure[1, -1] = self.right.value(Measure.pressureLast)
        self._velocity[1, -1] = -self.right.value(Measure.velocityMinusLast)
        self._friction[1, -1] = self.right.value(Measure.frictionLast)
        self._sos[1, -1] = self.right.value(Measure.BPspeedOfSoundLast)
        self._area[1] = self.right.value(Measure.area)
        self._pressure[2, -1] = self.right.value(Measure.pressureLast2)

        self._pressure[1, 0] = self.left.value(Measure.pressureLast)
        self._velocity[1, 0] = self.left.value(Measure.velocityPlusLast)
        self._friction[1, 0] = self.left.value(Measure.frictionLast)
        self._sos[1, 0] = self.left.value(Measure.BPspeedOfSoundLast)
        self._area[0] = self.left.value(Measure.area)
        self._pressure[2, 0] = self.left.value(Measure.pressureLast2)



    def prepare_next_inner_iteration(self, iteration: int) -> None:
        """
        Method to prepare the internal state for the next inner iteration of the current timestep

        :param iteration: Number of the next inner iteration to prepare for
        """
    def calculate_gravity_force(self, density):

        self._gravity_force = (((self._mat_density - density)
                                / self._mat_density)
                               * self._volume * (self._mat_density + density) * self._g)
        return None

    def calculate_spring_force(self):
        self._spring_force = self._prestressed_force + self._spring_stiffness * self._displacement[0, 1]
        return None


    def calculate_next_inner_iteration(self, iteration: int) -> bool:
        """
        Method to do the calculations of the next inner iteration

        :param iteration: Number of the next inner iteration
        :return: Whether this component needs another inner iteration afterwards
        """
        # Get the input fields
        pressure_a = self._pressure[1, 0]
        velocity_a = self._velocity[1, 0]
        friction_a = self._friction[1, 0]
        pressure_b = self._pressure[1, -1]
        velocity_b = self._velocity[1, -1]
        friction_b = self._friction[1, -1]
        area_a = self._area[0]
        area_b = self._area[1]
        sos_a = self._sos[1, 0]
        sos_b = self._sos[1, -1]

        # Calculate fluid properties
        density_a = self.fluid.density(pressure=pressure_a, temperature=None)
        density_b = self.fluid.density(pressure=pressure_b, temperature=None)



        # Perform actual calculation
        f1 = (density_a * sos_a * velocity_a
              + pressure_a
              - friction_a * self._delta_t * density_a * sos_a)

        f2 = (- density_b * sos_b * velocity_b
              + pressure_b
              + friction_b * self._delta_t * density_b*sos_b)

        k = area_a * area_a / (self._zeta * density_a)

        b = k*(density_a * sos_a / area_a
               + density_b * sos_b / area_b)
        c = k * (f1 - f2)

        self._volume_flow[0, 1] = np.sign(c) * (-b + np.sqrt(b * b + 2 * c))

        self._pressure[0, 1] = (density_a * sos_a * velocity_a
                                - density_a * sos_a * self._volume_flow[0, 1]/area_a
                                + pressure_a
                                - friction_a * self._delta_t * density_a)

        self._pressure[0, -1] = (density_b * sos_b * velocity_b
                                 - density_b * sos_b * self._volume_flow[0, 1] / area_b
                                 + pressure_b
                                 + friction_b * self._delta_t * density_b*sos_b)

        return False
