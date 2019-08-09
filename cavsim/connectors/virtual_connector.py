#! /opt/conda/bin/python3
""" Virtual Connector class for combining multiple connectors """

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


from typing import Tuple, Set
from ..measure import Measure
from .base_connector import BaseConnector
from .channel import Channel


class VirtualConnector(BaseConnector):
    """
    Virtual connector class to combine multiple other connectors
    """

    def __init__(self, connectors: Tuple[BaseConnector, ...]):
        """
        Initialization of the virtual connector class

        :param connectors: Tuple of connectors to be combine within this class
        :raises TypeError: Wrong type of at least one parameter
        :raises AssertionError: One of the connectors already assigned to another virtual connector
        :raises TypeError: Duplicate import/export channels
        """
        super(VirtualConnector, self).__init__()
        if not isinstance(connectors, tuple):
            raise TypeError('Wrong type for parameter connectors ({} != {})'.format(type(connectors), tuple))
        for connector in connectors:
            if not isinstance(connector, BaseConnector):
                # noinspection PyPep8
                raise TypeError('Wrong type for at least of element of parameter connectors ({} != {})'.format(type(connector), BaseConnector))  # pylint: disable=line-too-long
        # Test connectors for not being already in another virtual connector
        for connector in connectors:
            if connector._delegate is not None:  # pylint: disable=protected-access
                raise AssertionError('One of the connectors is already assigned to another virtual connector!')
        # Check for no duplicate channel types
        imports: Set[Measure] = set()
        exports: Set[Measure] = set()
        for connector in connectors:
            con_imports = connector.imports.union(connector.optionals)
            con_exports = connector.exports
            if not imports.isdisjoint(con_imports):
                raise TypeError('Duplicate import channels ({})!'.format(imports.intersection(con_imports)))
            if not exports.isdisjoint(con_exports):
                raise TypeError('Duplicate export channels ({})!'.format(exports.intersection(con_exports)))
            imports = imports.union(con_imports)
            exports = exports.union(con_exports)
        # Add the connectors to this class
        self._connectors: Tuple[BaseConnector, ...] = connectors
        for connector in connectors:
            connector._delegate = self  # pylint: disable=protected-access

    def _get_channels(self) -> Tuple[Channel, ...]:
        """
        Internal method to return a list of included channels

        :return: List of included channels
        """
        result: Tuple[Channel, ...] = ()
        for connector in self._connectors:
            result += connector._get_channels()  # pylint: disable=protected-access
        return result

    def release(self) -> None:
        """
        Method to release all connectors from the virtual connector
        """
        self.disconnect()
        for connector in self._connectors:
            connector._delegate = None  # pylint: disable=protected-access
        self._connectors = ()
