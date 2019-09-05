#! /opt/conda/bin/python3
""" Base component class """

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
from ..fluids.base_fluid import BaseFluid


class BaseComponent:
    """
    Abstract base component class to define the required interface methods
    """

    def __init__(self) -> None:
        """
        Initialization of the class
        """
        self._fluid: Optional[BaseFluid] = None
        self._global_fluid: Optional[BaseFluid] = None

    @property
    def fluid(self) -> BaseFluid:
        """
        Fluid property of the component (local or global)

        :return: Fluid used for this component
        :raises AssertionError: Neither local nor global fluid set
        """
        if self._fluid is not None:
            return self._fluid
        if self._global_fluid is not None:
            return self._global_fluid
        raise AssertionError('Cannot find any fluid assigned to the component!')

    @fluid.setter
    def fluid(self, fluid: Optional[BaseFluid]) -> None:
        """
        Setter method for local fluid property of the component

        :param fluid: New value for local fluid property or None for global
        :raises TypeError: Assigned value is neither None nor a BaseFluid class
        """
        if fluid is not None and not isinstance(fluid, BaseFluid):
            raise TypeError('Wrong type of assigned value ({} != {})'.format(type(fluid), BaseFluid))
        self._fluid = fluid

    # noinspection PyMethodMayBeStatic
    def get_max_delta_t(self) -> Optional[float]:  # pylint: disable=no-self-use, redundant-returns-doc
        """
        Method to return the maximum allowed timestep width for this component

        :return: Maximum allowed timestep width or None if any is suitable
        """
        return None

    def check_fluid(self, global_fluid: BaseFluid) -> None:
        """
        Method to check the assigned fluid (or use the global fluid)

        :param global_fluid: Global fluid to be used if no local fluid is assigned
        """
        self._global_fluid = global_fluid

    def discretize(self, delta_t: float) -> None:
        """
        Method handling the discretization of the component (for a given timestep width)

        :param delta_t: Timestep width to discretize for
        """

    def initialize(self) -> None:
        """
        Initialize the internal state of the component (after discretization was called)
        """

    def prepare_next_timestep(self, delta_t: float, next_total_time: float) -> None:
        """
        Prepare the internal state for the next timestep to be calculated

        :param delta_t: Timestep width for the next timestep
        :param next_total_time: Total simulation time at the end of the next timestep
        """

    def exchange_last_boundaries(self) -> None:
        """
        Exchange the boundary values from previous time steps
        """

    def prepare_next_inner_iteration(self, iteration: int) -> None:
        """
        Method to prepare the internal state for the next inner iteration of the current timestep

        :param iteration: Number of the next inner iteration to prepare for
        """

    def exchange_current_boundaries(self) -> None:
        """
        Exchange the boundary values from the current time steps
        :return:
        """

    def finalize_current_timestep(self) -> None:
        pass

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def calculate_next_inner_iteration(self, iteration: int) -> bool:  # pylint: disable=no-self-use, unused-argument
        """
        Method to do the calculations of the next inner iteration

        :param iteration: Number of the next inner iteration
        :return: Whether this component needs another inner iteration afterwards
        """
        return False
