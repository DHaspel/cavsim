#! /opt/conda/bin/python3
""" Base solver class containing general methods for solvers """

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


from typing import List, Union
from ..components.component import Component


class BaseSolver:
    """
    Base solver class containing general methods for derived solvers
    """

    def __init__(self) -> None:
        """
        Initialization of the class
        """
        self._seeds: List[Component] = []

    @property
    def disconnected(self) -> bool:
        """
        Property whether there are disconnected components in the system

        :return: Whether some components are disconnected
        """
        try:
            self._get_connected_list(self._seeds, True)
        except AssertionError:
            return True
        return False

    @property
    def seeds(self) -> List[Component]:
        """
        System seed components property

        :return: List of system seeds
        """
        return self._seeds

    @seeds.setter
    def seeds(self, seeds: Union[Component, List[Component]]) -> None:
        """
        Setter for system seed components property

        :param seeds: Component or list of components to be set as seeds
        :raises TypeError: Wrong type of assigned value
        """
        if isinstance(seeds, Component):
            self._seeds = [seeds]
        elif isinstance(seeds, list):
            for seed in seeds:
                if not isinstance(seed, Component):
                    # noinspection PyPep8
                    raise TypeError('Wrong type of one entry in assigned value list ({} != {})'.format(type(seed), Component))
            self._seeds = seeds
        else:
            raise TypeError('Wrong type of assigned value ({} != {} or {})'.format(type(seeds), Component, list))

    @staticmethod
    def _get_connected(seed: Component, exception_on_disconnected: bool = True) -> List[Component]:
        """
        Method to return all connected components starting from a seed component

        :param seed: Starting component to search from
        :param exception_on_disconnected: Whether to raise an exception on disconnected connector ends
        :return: List of all connected components
        :raises AssertionError: Disconnected component within the system
        :raises TypeError: Wrong base type of one of the systems components
        """
        result: List[Component] = [seed]
        for component in result:
            for connector in component.connectors:
                if connector.connected is False and exception_on_disconnected is True:
                    raise AssertionError('Found at least one unconnected component! {}'.format(component))
                if connector.connected is True and connector.link is not None:
                    neighbour = connector.link
                    for next_component in neighbour.components:
                        if not isinstance(next_component, Component):
                            # noinspection PyPep8
                            raise TypeError('Found component class ({}) is not derived from {}!'.format(type(next_component), Component))  # pylint: disable=line-too-long
                        if next_component not in result:
                            result.append(next_component)
        return result

    def _get_connected_list(self, seeds: List[Component], exception_on_disconnected: bool = True) -> List[Component]:
        """
        Method to return all connected components starting from a list of seeds

        :param seeds: List of starting component to search from
        :param exception_on_disconnected: Whether to raise an exception on disconnected connector ends
        :return: List of all connected components
        """
        result: List[Component] = []
        for seed in seeds:
            components = self._get_connected(seed, exception_on_disconnected)
            for component in components:
                if component not in result:
                    result.append(component)
        return result

    @property
    def components(self) -> List[Component]:
        """
        List of all connected components property

        :return: List of all components connected to the seeds
        """
        return self._get_connected_list(self._seeds, False)
