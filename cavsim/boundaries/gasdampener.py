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


class GasDampener(BaseBoundary):
    """
    Pipe class implementing the pipe simulation calculations
    """

    def __init__(
            self,
            pressure0: float,
            volume0: float,
            polytropic_exponent: float,
            zeta: float,
    ) -> None:
        """
        Initialization of the class

        :param velocity: Given velocity at the boundary [m/s]
        :raises TypeError: Wrong type of at least one parameter
        :raises ValueError: Value of at least one parameter out of bounds
        """
        super(GasDampener, self).__init__()

        # Register internal fields

        self._pressure: np.ndarray = self.field_create('pressure', 3)
        self._velocity: np.ndarray = self.field_create('velocity', 3)
        self._volume_flow: np.ndarray = self.field_create('volume_flow', 3)
        self._sos: np.ndarray = self.field_create('speed_of_sound', 3)
        self._friction = self.field_create('friction', 3)
        self._area = np.empty(2)
        self._pressure0 = pressure0
        self._volume0 = volume0
        self._polytropic_exponent = polytropic_exponent
        self._zeta = zeta
        self._volume: np.ndarray = self.field_create('volume', 3)

        # Create the left connector
        self._left: Connector = Connector(self, [
            ImportChannel(Measure.deltaX, False),
            ExportChannel(Measure.boundaryPoint, lambda: True),
            ImportChannel(Measure.pressureLast, False),
            ImportChannel(Measure.pressureLast2, False),
            ExportChannel(Measure.pressureCurrent, lambda: self._pressure[0, -2]),
            ExportChannel(Measure.pressureLast, lambda: self._pressure[1, -2]),
            ImportChannel(Measure.velocityPlusLast, False),
            ExportChannel(Measure.velocityMinusCurrent,
                          lambda: -self._volume_flow[0, 0]/(self._area[0])),
            ExportChannel(Measure.velocityMinusLast,
                          lambda: -self._volume_flow[1, 0]/(self._area[0])),
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
                          lambda: self._volume_flow[0, -1]/(self._area[1])),
            ExportChannel(Measure.velocityPlusLast,
                          lambda: self._volume_flow[1, -1]/(self._area[1])),
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
    def pressure_init(self) -> float:
        """

        :return:
        """
        return self._pressure_init

    @property
    def right(self) -> Connector:
        """
        Right connector property

        :return: Right sided connector of the pipe
        """
        return self._right

    @property
    def polytropic_exponent(self) -> float:
        """
        Polytropic exponent of the compression property

        :return: Polytropic exponent of the compression
        """
        return self._polytropic_exponent

    @property
    def pressure0(self) -> float:
        """
        Initial pressure of the gas bubble property

        :return: Initial pressure of the gas bubble
        """
        return self._pressure0

    @property
    def volume0(self) -> float:
        """
        Initial volume of the gas bubble property

        :return: Initial volume of the gas bubble
        """
        return self._volume0

    @property
    def zeta(self)-> float:
        """
        Dimensionless friction value property

        :return: Dimensionless friction value (zeta-value)r
        """
        return self._zeta

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
        self.field('volume')[:, :] = self._volume0 * np.ones(self.field('volume').shape)[:, :]

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
        volume = self._volume[1, 1]
        volume_flow = self._volume_flow[1, 1]

        # Calculate fluid properties
        density_a = self.fluid.density(pressure=pressure_a, temperature=None)
        density_b = self.fluid.density(pressure=pressure_b, temperature=None)

        # Perform actual calculation

        # Simplify factors
        f1 = (area_a * velocity_a
              + area_a/(sos_a * density_a) * pressure_a
              - area_a * friction_a * self._delta_t)

        f2 = (area_b * velocity_b
              - area_b / (sos_b * density_b) * pressure_b
              - area_b * friction_b * self._delta_t)

        f3 = (-area_a / (sos_a * density_a)
              - area_b / (sos_b * density_b))

        f4 = f1 - f2

        # Initializing newton's iteration

        epsilon = 1
        # First guess -> former time step

        volume_flow = self._volume_flow[1, 1]
        # Iteration start

        while epsilon > 1e-7:

            vol_fun = ((- f4 / f3 + volume_flow / f3
                        - self._zeta * (density_a * np.abs(volume_flow) * volume_flow / (2 * area_a * area_a)))
                       * np.power((volume - volume_flow * self._delta_t), self._polytropic_exponent)
                       - self._pressure0 * np.power(self._volume0, self._polytropic_exponent))

            dvol_fun = (self._polytropic_exponent
                        * (- f4 / f3 + volume_flow / f3)
                        * np.power((volume - volume_flow * self._delta_t), (self._polytropic_exponent - 1))
                        * (- self._delta_t)
                        + (1.0 / f3 + self._zeta * density_a * np.abs(volume_flow) /(area_a * area_a))
                        * np.power((volume - volume_flow * self._delta_t), self._polytropic_exponent))

            new_volume_flow = volume_flow - vol_fun / dvol_fun
            epsilon = new_volume_flow - volume_flow
            volume_flow = new_volume_flow

        self._volume_flow[0, 1] = volume_flow
        self._volume[0, 1] = volume - volume_flow * self._delta_t

        self._pressure[0, 1] = (volume_flow - f4) / f3

        self._volume_flow[0, 0] = - area_a / (density_a * sos_a) * self._pressure[0, 1] + f1
        self._volume_flow[0, -1] = area_b / (density_b * sos_b) * self._pressure[0, 1] + f2

        return False
