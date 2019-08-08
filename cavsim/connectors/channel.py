#! /opt/conda/bin/python3
""" Base class for connectors data channels """

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
from ..measure import Measure


class Channel:
    """
    Channel base class for transferring measures between two connected channels
    """

    def __init__(self, measure: Measure, is_import: bool, optional: bool = False) -> None:
        """
        Initialization of the channel class

        :param measure: Measure unit which is transferred via the channel
        :param is_import: Whether the transfer is importing (or exporting)
        :param optional: Whether the transfer channel is optional (for importing channels)
        """
        super(Channel, self).__init__()
        self._measure: Measure = measure
        self._connection: Optional['Channel'] = None
        self._is_import: bool = is_import
        self._optional: bool = optional

    @property
    def measure(self) -> Measure:
        """
        Measure unit of the channel

        :return: Measure unit of the channel
        """
        return self._measure

    @property
    def connected(self) -> bool:
        """
        Connection state of the channel

        :return: Whether the channel is currently connected to another channel
        """
        return self._connection is not None

    @property
    def optional(self) -> bool:
        """
        Optional state of the channel

        :return: Whether the channel is optional
        """
        return self._optional

    @property
    def is_import(self) -> bool:
        """
        Import state of the channel

        :return: Whether the channel direction is importing
        """
        return self._is_import

    @property
    def is_export(self) -> bool:
        """
        Export state of the channel

        :return: Whether the channel direction is exporting
        """
        return not self._is_import

    def is_valid_connection(self, channel: 'Channel') -> bool:
        """
        Checks if a connection with another channel is valid

        :param channel: Other channel to connect to
        :return: Whether the connection is valid
        """
        measure = (channel.measure == self.measure)
        direction = (channel.is_export == self.is_import)
        connected = not channel.connected and not self.connected
        return measure and direction and connected

    def connect(self, channel: 'Channel') -> None:
        """
        Connect to another channel

        :param channel: Other channel to connect to
        :raises TypeError: Mismatching channels (measure unit or channel directions)
        :raises AssertionError: One of the channels is already connected
        """
        if self.measure is not channel.measure:
            raise TypeError('Channels have different measures ({}, {})'.format(self.measure, channel.measure))
        if self.is_import == channel.is_import:
            raise TypeError('Channels can only connect one import to one export channel!')
        if self.connected or channel.connected:
            raise AssertionError('Channels can only be connected if both are currently disconnected!')
        self._connection = channel
        channel._connection = self  # pylint: disable=protected-access

    def disconnect(self) -> None:
        """
        Disconnects the channel (if it was connected)
        """
        if self._connection is not None:
            self._connection._connection = None  # pylint: disable=protected-access
        self._connection = None

    def __del__(self) -> None:
        """
        Destructor of the channel class
        """
        self.disconnect()
