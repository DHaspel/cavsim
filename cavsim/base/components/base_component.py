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


class BaseComponent:
    """
    Abstract base component class to define the required interface methods
    """

    def __init__(self) -> None:
        """
        Initialization of the class
        """

    # noinspection PyMethodMayBeStatic
    def get_max_delta_t(self) -> Optional[float]:  # pylint: disable=no-self-use, redundant-returns-doc
        """
        Method to return the maximum allowed timestep width for this component

        :return: Maximum allowed timestep width or None if any is suitable
        """
        return None

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

    def prepare_next_inner_iteration(self, iteration: int) -> None:
        """
        Method to prepare the internal state for the next inner iteration of the current timestep

        :param iteration: Number of the next inner iteration to prepare for
        """

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def calculate_next_inner_iteration(self, iteration: int) -> bool:  # pylint: disable=no-self-use, unused-argument
        """
        Method to do the calculations of the next inner iteration

        :param iteration: Number of the next inner iteration
        :return: Whether this component needs another inner iteration afterwards
        """
        return False
