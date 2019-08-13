#! /opt/conda/bin/python3
""" Component class to derive all components from """

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


from typing import List, Union, Tuple, Optional
from .base_component import BaseComponent
from ..connectors.base_connector import BaseConnector


class Component(BaseComponent):
    """
    Component class to be used to inherit real components from
    """

    def __init__(self) -> None:
        """
        Initialization of the class
        """
        super(Component, self).__init__()
        self._connectors: List[BaseConnector] = []

    @property
    def connectors(self) -> List[BaseConnector]:
        """
        Property: List of all connectors of the component

        :return: List of connectors of the component
        """
        return self._connectors

    @property
    def connector_count(self) -> int:
        """
        Number of total connectors property

        :return: Total number of connectors of the component
        """
        return len(self._connectors)

    @property
    def connected(self) -> int:
        """
        Number of connected connectors property

        :return: Number of currently connected connectors of the component
        """
        count = 0
        for connector in self._connectors:
            if connector.connected is True:
                count += 1
        return count

    @property
    def unconnected(self) -> int:
        """
        Number of unconnected connectors property

        :return: Number of currently unconnected connectors of the component
        """
        count = 0
        for connector in self._connectors:
            if connector.connected is False:
                count += 1
        return count

    def disconnect(self) -> None:
        """
        Method to disconnect all connectors of the component
        """
        for connector in self._connectors:
            connector.disconnect()

    def connect(self, connect_to: Union['Component', BaseConnector]) -> None:
        """
        Method to connect the component to another component or connector

        :param connect_to: Component of connector to connect to
        :raises TypeError: Wrong type of at least one parameter
        :raises AssertionError: Connection failed for either no possible connections or no unique combination detectable
        """
        if not isinstance(connect_to, BaseConnector) and not isinstance(connect_to, Component):
            # noinspection PyPep8
            raise TypeError('Wrong type for parameter connect_to ({} != {} or {})'.format(type(connect_to), BaseConnector, Component))  # pylint: disable=line-too-long
        # Get available connectors of both ends
        sources = [connector for connector in self.connectors if connector.connected is False]
        if isinstance(connect_to, BaseConnector):
            targets = [connect_to]
        else:
            targets = [connector for connector in connect_to.connectors if connector.connected is False]
        # Find matches
        match: Optional[Tuple[BaseConnector, BaseConnector]] = None
        match_count = 0
        for source in sources:
            for target in targets:
                if source.connectable(target):
                    match = (source, target)
                    match_count += 1
        if match_count > 1:
            raise AssertionError('Too many possible combinations ({}) for the connection!'.format(match_count))
        if match is None:
            raise AssertionError('No valid connection combination found!')
        source, target = match
        source.connect(target)

    def _add_connector(self, new_connector: BaseConnector) -> None:
        if not isinstance(new_connector, BaseConnector):
            # noinspection PyPep8
            raise TypeError('Wrong type for parameter new_connector ({} != {})'.format(type(new_connector), BaseConnector))
        valid = True
        for connector in self.connectors:
            if connector == new_connector:
                valid = False
        if valid is True:
            self._connectors.append(new_connector)
