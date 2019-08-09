#! /opt/conda/bin/python3
""" Connector class for a set of data channels """

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
from .channel import Channel
from .base_connector import BaseConnector


class Connector(BaseConnector):
    """
    Connector class for connecting a sets of channels
    """

    def __init__(self, channels: Tuple[Channel, ...]) -> None:
        """
        Initialization of the connector class

        :param channels: Tuple of channels to be included in the connector
        :raises TypeError: Wrong type of at least one parameter
        :raises ValueError: Duplicate channels for at least one measure
        """
        super(Connector, self).__init__()
        if not isinstance(channels, tuple):
            raise TypeError('Wrong type for parameter channels ({} != {})'.format(type(channels), tuple))
        for channel in channels:
            if not isinstance(channel, Channel):
                # noinspection PyPep8
                raise TypeError('Wrong type for element in parameter channels ({} != {})'.format(type(channel), Channel))
        # Check for duplication of channels
        set_in: Set[Measure] = set()
        set_out: Set[Measure] = set()
        for channel in channels:
            already = (channel.measure in set_in) if channel.is_import is True else (channel.measure in set_out)
            if already is True:
                # noinspection PyPep8
                raise ValueError('Duplicate channel ({}) for {} direction!'.format(channel.measure, 'import' if channel.is_import is True else 'export'))  # pylint: disable=line-too-long
            if channel.is_import is True:
                set_in.add(channel.measure)
            else:
                set_out.add(channel.measure)
        # Set the internal channels tuple
        self._channels: Tuple[Channel, ...] = channels

    def _get_channels(self) -> Tuple[Channel, ...]:
        """
        Internal method to return a list of included channels

        :return: List of included channels
        """
        return self._channels
