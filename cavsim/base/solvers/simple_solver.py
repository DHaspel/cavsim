#! /opt/conda/bin/python3
""" Simple solver class implementation """

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


from warnings import warn
from typing import Optional
from .base_solver import BaseSolver
from ..progressbar import ProgressBar


class SimpleSolver(BaseSolver):
    """
    Simple solver class
    """

    def _get_delta_t(self, delta_t: float) -> float:
        """
        Internal method to check for delta_t limitations of the components

        :param delta_t: Desired delta_t from user
        :return: Resulting delta_t respecting component limits
        """
        timestep: float = delta_t
        for component in self.components:
            component.check_fluid(self._fluid)
        for component in self.components:
            component_time = component.get_max_delta_t()
            if component_time is not None:
                timestep = min(timestep, component_time)
                if component_time < delta_t:
                    # noinspection PyPep8
                    warn('Smaller timestep required by component! ({} < {} by {})'.format(component_time, delta_t, component))
        return timestep

    def _discretize(self, delta_t: float) -> None:
        """
        Internal method to discretize and initialize all components

        :param delta_t: Delta_t used for the simulation
        """
        for component in self.components:
            component.discretize(delta_t)
        for component in self.components:
            component.initialize()

    def _solve_inner_loop(self, max_iterations: Optional[int] = None) -> None:
        """
        Internal method to perform the inner solving loop

        :param max_iterations: Maximum number of allowed inner iterations
        """
        need_inner: bool = True
        inner_count: int = 0
        while need_inner:
            need_inner = False
            for component in self.components:
                component.prepare_next_inner_iteration(inner_count)
            for component in self.components:
                component.exchange_current_boundaries()
            for component in self.components:
                need_inner = need_inner or component.calculate_next_inner_iteration(inner_count)
            inner_count += 1
            if isinstance(max_iterations, int) and inner_count >= max_iterations and need_inner is True:
                warn('Limit of maximum inner iterations exceeded! ({})'.format(max_iterations))
                break

    def solve(  # pylint: disable=too-complex
            self,
            delta_t: float,
            total_time: float,
            max_iterations: Optional[int] = None,
            verbosity: int = 1
    ) -> None:
        """
        Method to solve all components with given timestep width and for given total time

        :param delta_t: Timestep width to use for solving
        :param total_time: Total time to solve for
        :param max_iterations: Maximum number of allowed inner iterations
        :param verbosity: Verbosity of the return information
        :raises TypeError: Wrong type of at least one parameter
        """
        # Validate parameters
        if not isinstance(delta_t, float) or delta_t < 0.0:
            raise TypeError('Wrong type for parameter delta_t ({} != {})'.format(type(delta_t), float))
        if not isinstance(total_time, float) or total_time < 0.0:
            raise TypeError('Wrong type for parameter total_time ({} != {})'.format(type(total_time), float))
        if max_iterations is not None and not isinstance(max_iterations, int):
            raise TypeError('Wrong type for parameter max_iterations ({} != {})'.format(type(max_iterations), int))
        if not isinstance(verbosity, int) or verbosity < 0 or verbosity > 2:
            raise TypeError('Wrong type for parameter verbosity ({} != {})'.format(type(verbosity), int))
        # Perform the solving
        delta_t = self._get_delta_t(delta_t)
        self._discretize(delta_t)
        current_time: float = 0.0
        progress = ProgressBar()
        while current_time < total_time:
            current_time += delta_t
            for component in self.components:
                component.prepare_next_timestep(delta_t, current_time)
            for component in self.components:
                component.exchange_last_boundaries()
            self._solve_inner_loop(max_iterations)
            for component in self.components:
                component.finalize_current_timestep()
            if self._callback is not None:
                self._callback()
            if verbosity > 0:
                progress.update(
                    current_time / total_time,
                    'Currently at time {:7.3f} of {:7.3f}'.format(current_time, total_time)
                )
