#! /opt/conda/bin/python3
""" Base connector class for a set of data channels """

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


from typing import Set, Optional, List
from ..measure import Measure
from .channel import Channel
from ..components.base_component import BaseComponent


class BaseConnector:
    """
    Base connector class for connecting a sets of channels
    """

    def __init__(self) -> None:
        """
        Initialization of the bae connector class
        """
        super(BaseConnector, self).__init__()
        self._delegate: Optional['BaseConnector'] = None
        self._link: Optional['BaseConnector'] = None

    @property
    def delegate(self) -> Optional['BaseConnector']:
        """
        Property which indicates the container the connector is assigned to

        :return: Delegated container of the connector
        """
        return self._delegate

    @property
    def link(self) -> Optional['BaseConnector']:
        """
        Property referencing the connector this one is connected to

        :return: Connected connector or None
        """
        if self.delegate is not None:
            return self.delegate.link
        return self._link

    @property
    def connected(self) -> bool:
        """
        Property indicating the connection state of the class

        :return: Whether the class is currently connected
        :raises ValueError: Inconsistent partially connected state
        """
        if self.delegate is not None:
            return self.delegate.connected
        any_connected = False
        miss_required = False
        for channel in self.channels:
            if channel.connected is True:
                any_connected = True
            elif channel.is_import is True and channel.optional is False:
                miss_required = True
        if any_connected is True and miss_required is True:
            raise ValueError('Connector is only partially connected and in an inconsistent state!')
        return any_connected

    @property
    def channels(self) -> List[Channel]:
        """
        List of all included channels property

        :return: List of included channels
        """
        if self.delegate is not None:
            return self.delegate.channels
        return self._get_channels()

    def _get_channels(self) -> List[Channel]:  # pylint: disable=no-self-use
        """
        Internal method to return a list of included channels

        :return: List of included channels
        """
        return []

    @property
    def components(self) -> List[BaseComponent]:
        """
        List of all assigned components property

        :return: List of assigned components
        """
        if self.delegate is not None:
            return self.delegate.components
        return self._get_components()

    def _get_components(self) -> List[BaseComponent]:  # pylint: disable=no-self-use
        """
        Internal method to return a list of assigned components

        :return: List of assigned components
        """
        return []

    @property
    def imports(self) -> Set[Measure]:
        """
        Set of required import measures property

        :return: Set of measures being required imports
        """
        if self.delegate is not None:
            return self.delegate.imports
        result = set()
        for channel in self.channels:
            if channel.is_import is True and channel.optional is False:
                result.add(channel.measure)
        return result

    @property
    def optionals(self) -> Set[Measure]:
        """
        Set of optional import measures property

        :return: Set of measures being optional imports
        """
        if self.delegate is not None:
            return self.delegate.optionals
        result = set()
        for channel in self.channels:
            if channel.is_import is True and channel.optional is True:
                result.add(channel.measure)
        return result

    @property
    def exports(self) -> Set[Measure]:
        """
        Set of export measures property

        :return: Set of measures being exports
        """
        if self.delegate is not None:
            return self.delegate.exports
        result = set()
        for channel in self.channels:
            if channel.is_import is False:
                result.add(channel.measure)
        return result

    def connectable(self, connector: 'BaseConnector') -> bool:
        """
        Method to check whether the class can be connected to another connector

        :param connector: Connect which should be tested for allowing connection
        :return: Whether a connection is possible
        :raises TypeError: Wrong type of at least one parameter
        """
        if self.delegate is not None:
            return self.delegate.connectable(connector)
        if not isinstance(connector, BaseConnector):
            raise TypeError('Wrong type for parameter connector ({} != {})'.format(type(connector), BaseConnector))
        unconnected = (self.connected is False) and (connector.connected is False)
        match1 = self.imports.issubset(connector.exports)
        match2 = connector.imports.issubset(self.exports)
        valid = self.delegate is None and connector.delegate is None
        return unconnected and match1 and match2 and valid

    def disconnect(self) -> None:
        """
        Method to terminate a previous connection
        """
        if self.delegate is not None:
            self.delegate.disconnect()
        elif self.connected:
            for channel in self.channels:
                channel.disconnect()
            if self._link is not None:
                self._link._link = None  # pylint: disable=protected-access
            self._link = None

    def connect(self, connector: 'BaseConnector') -> None:
        """
        Method to connect this class to another connector

        :param connector: Another connector to connect to
        :raises TypeError: Wrong type of at least one parameter
        :raises AssertionError: One of the connectors already connected
        :raises TypeError: Channels of the connectors are not matching
        """
        if self.delegate is not None:
            self.delegate.connect(connector)
            return None
        if not isinstance(connector, BaseConnector):
            raise TypeError('Wrong type for parameter connector ({} != {})'.format(type(connector), BaseConnector))
        if connector.delegate is not None:
            raise AssertionError('Connectors can only be connected if neither is part of a container!')
        if self.connected or connector.connected:
            raise AssertionError('Connectors can only be connected if both are currently disconnected!')
        match1 = self.imports.issubset(connector.exports)
        match2 = connector.imports.issubset(self.exports)
        if not (match1 and match2):
            raise TypeError('Connection impossible: channels are not matching!')
        # Apply the connections in both directions (including optionals)
        for source, target in ((self.channels, connector.channels), (connector.channels, self.channels)):
            exports = {channel.measure: channel for channel in target if channel.is_import is False}
            # noinspection PyPep8
            imports = {channel.measure: channel for channel in source if channel.is_import is True and channel.optional is False}  # pylint: disable=line-too-long
            # noinspection PyPep8
            optionals = {channel.measure: channel for channel in source if channel.is_import is True and channel.optional is True}  # pylint: disable=line-too-long
            for channel in imports.values():
                channel.connect(exports[channel.measure])
            for channel in optionals.values():
                if channel.measure in exports.keys():
                    channel.connect(exports[channel.measure])
        connector._link = self  # pylint: disable=protected-access
        self._link = connector
        return None
