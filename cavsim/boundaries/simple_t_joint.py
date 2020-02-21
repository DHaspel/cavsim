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


class SimpleTJoint(BaseBoundary):
    """
    Pipe class implementing the pipe simulation calculations
    """

    def __init__(
            self,
            initial_pressure: float = None,
    ) -> None:
        """
        Initialization of the class

        :param velocity: Given velocity at the boundary [m/s]
        :raises TypeError: Wrong type of at least one parameter
        :raises ValueError: Value of at least one parameter out of bounds
        """
        super(SimpleTJoint, self).__init__()

        # Register internal fields

        self._pressure: np.ndarray = self.field_create('pressure', 4)
        self._velocity: np.ndarray = self.field_create('velocity', 4)
        self._volume_flow: np.ndarray = self.field_create('volume_flow', 4)
        self._sos: np.ndarray = self.field_create('speed_of_sound', 4)
        self._friction = self.field_create('friction', 4)
        self._area = np.ones(3)
        self._initial_pressure = initial_pressure

        # Create the left connector
        self._left: Connector = Connector(self, [
            ImportChannel(Measure.deltaX, False),
            ExportChannel(Measure.boundaryPoint, lambda: True),
            ImportChannel(Measure.pressureLast, False),
            ImportChannel(Measure.pressureLast2, False),
            ExportChannel(Measure.pressureCurrent, lambda: self._pressure[0, 1]),
            ExportChannel(Measure.pressureLast, lambda: self._pressure[1, 1]),
            ImportChannel(Measure.velocityPlusLast, False),
            ExportChannel(Measure.velocityMinusCurrent,
                          lambda: -self._volume_flow[0, 0]/(self._area[0]
                                                            * self.fluid.density(pressure=self._pressure[1, 0],
                                                                                 temperature=None))),
            ExportChannel(Measure.velocityMinusLast,
                          lambda: -self._volume_flow[1, 0]/(self._area[0]
                                                            * self.fluid.density(pressure=self._pressure[2, 0],
                                                                                 temperature=None))),
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
            ExportChannel(Measure.pressureCurrent, lambda: self._pressure[0, 1]),
            ExportChannel(Measure.pressureLast, lambda: self._pressure[1, 1]),
            ImportChannel(Measure.velocityMinusLast, False),
            ExportChannel(Measure.velocityPlusCurrent,
                          lambda: self._volume_flow[0, -2]/(self._area[1]
                                                            * self.fluid.density(pressure=self._pressure[1, -2],
                                                                                 temperature=None))),
            ExportChannel(Measure.velocityPlusLast,
                          lambda: self._volume_flow[1, -2]/(self._area[1]
                                                            * self.fluid.density(pressure=self._pressure[2, -2],
                                                                                 temperature=None))),
            ImportChannel(Measure.frictionLast, False),
            ImportChannel(Measure.BPspeedOfSoundLast, False),
            ImportChannel(Measure.area)
        ])

        self._right2: Connector = Connector(self, [
            ImportChannel(Measure.deltaX, False),
            ExportChannel(Measure.boundaryPoint, lambda: True),
            ImportChannel(Measure.pressureLast, False),
            ImportChannel(Measure.pressureLast2, False),
            ExportChannel(Measure.pressureCurrent, lambda: self._pressure[0, 1]),
            ExportChannel(Measure.pressureLast, lambda: self._pressure[1, 1]),
            ImportChannel(Measure.velocityMinusLast, False),
            ExportChannel(Measure.velocityPlusCurrent,
                          lambda: self._volume_flow[0, -1]/(self._area[2]
                                                            * self.fluid.density(pressure=self._pressure[1, -1],
                                                                                 temperature=None))),
            ExportChannel(Measure.velocityPlusLast,
                          lambda: self._volume_flow[1, -1]/(self._area[2]
                                                            * self.fluid.density(pressure=self._pressure[2, -1],
                                                                                 temperature=None))),
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
    def right2(self) -> Connector:
        """
        Right connector property

        :return: Right sided connector of the pipe
        """
        return self._right2

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

    def discretize(self, delta_t: float) -> None:
        """
        Method handling the discretization of the component (for a given timestep width)

        :param delta_t: Timestep width to discretize for
        :raises ValueError: Timestep too large to fit at least 3 inner points
        """
        self._delta_t = delta_t
        self.fields_resize(4)

    def initialize(self) -> None:
        """
        Initialize the internal state of the component (after discretization was called)
        """

        if self.initial_pressure is not None:
            self.field('pressure')[:, :] = self.initial_pressure * np.ones(self.field('pressure').shape)[:, :]
        else:
            self.field('pressure')[:, :] = self.fluid.initial_pressure * np.ones(self.field('pressure').shape)[:, :]

        self.field('velocity')[:, :] = np.zeros(self.field('velocity').shape)[:, :]
        #self.field('pressure')[:, :] = self.fluid.initial_pressure * np.ones(self.field('pressure').shape)[:, :]
        self.field('friction')[:, :] = np.zeros(self.field('friction').shape)[:, :]
        self.field('speed_of_sound')[:, :] = np.zeros(self.field('friction').shape)[:, :]

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

        self._pressure[1, 0] = self.left.value(Measure.pressureLast)
        self._velocity[1, 0] = self.left.value(Measure.velocityPlusLast)
        self._friction[1, 0] = self.left.value(Measure.frictionLast)
        self._sos[1, 0] = self.left.value(Measure.BPspeedOfSoundLast)
        self._area[0] = self.left.value(Measure.area)
        self._pressure[2, 0] = self.left.value(Measure.pressureLast2)

        # Exchange previous values with the first right boundary

        self._pressure[1, -2] = self.right.value(Measure.pressureLast)
        self._velocity[1, -2] = -self.right.value(Measure.velocityMinusLast)
        self._friction[1, -2] = self.right.value(Measure.frictionLast)
        self._sos[1, -2] = self.right.value(Measure.BPspeedOfSoundLast)
        self._area[1] = self.right.value(Measure.area)
        self._pressure[2, -2] = self.right.value(Measure.pressureLast2)

        # Exchange previous values with the second right boundary

        self._pressure[1, -1] = self.right2.value(Measure.pressureLast)
        self._velocity[1, -1] = -self.right2.value(Measure.velocityMinusLast)
        self._friction[1, -1] = self.right2.value(Measure.frictionLast)
        self._sos[1, -1] = self.right2.value(Measure.BPspeedOfSoundLast)
        self._area[2] = self.right2.value(Measure.area)
        self._pressure[2, -1] = self.right2.value(Measure.pressureLast2)



    def prepare_next_inner_iteration(self, iteration: int) -> None:
        """
        Method to prepare the internal state for the next inner iteration of the current timestep

        :param iteration: Number of the next inner iteration to prepare for
        """

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

        pressure_b = self._pressure[1, -2]
        velocity_b = self._velocity[1, -2]
        friction_b = self._friction[1, -2]

        pressure_c = self._pressure[1, -1]
        velocity_c = self._velocity[1, -1]
        friction_c = self._friction[1, -1]

        area_a = self._area[0]
        area_b = self._area[1]
        area_c = self._area[2]

        sos_a = self._sos[1, 0]
        sos_b = self._sos[1, -2]
        sos_c = self._sos[1, -1]

        # Calculate fluid properties

        density_a = self.fluid.density(pressure=pressure_a, temperature=None)
        density_b = self.fluid.density(pressure=pressure_b, temperature=None)
        density_c = self.fluid.density(pressure=pressure_c, temperature=None)

        # Perform actual calculation for the current pressure

        f1 = (density_a * area_a * velocity_a
              + area_a * pressure_a / sos_a
              - density_a * area_a * self._delta_t * friction_a)

        f2 = (density_b * area_b * velocity_b
              - area_b * pressure_b / sos_b
              - density_b * area_b * self._delta_t * friction_b)

        f3 = (density_c * area_c * velocity_c
              - area_c * pressure_c / sos_c
              - density_c * area_c * self._delta_t * friction_c)

        result = ((f1 - f2 - f3)
                  / (area_a / sos_a
                     + area_b / sos_b
                     + area_c / sos_c))

        self._pressure[0, 1] = result

        # Perform actual calculation for the current velocity

        self._volume_flow[0, 0] = (density_a * area_a * velocity_a
                                   + (pressure_a - result) * area_a / sos_a
                                   - density_a * area_a * self._delta_t * friction_a)

        self._volume_flow[0, -2] = (density_b * area_b * velocity_b
                                    + (result - pressure_b) * area_b / sos_b
                                    - density_b * area_b * self._delta_t * friction_b)

        self._volume_flow[0, -1] = (density_c * area_c * velocity_c
                                    + (result - pressure_c) * area_c / sos_c
                                    - density_c * area_c * self._delta_t * friction_c)

        return False
