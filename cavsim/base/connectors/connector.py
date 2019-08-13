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


from typing import Set, List, Optional
from ...measure import Measure
from ..channels.base_channel import BaseChannel
from .base_connector import BaseConnector
from ..components.base_component import BaseComponent


class Connector(BaseConnector):
    """
    Connector class for connecting a sets of channels
    """

    def __init__(self, parent: Optional[BaseComponent], channels: List[BaseChannel]) -> None:
        """
        Initialization of the connector class

        :param parent: Component the connector is assigned to
        :param channels: List of channels to be included in the connector
        :raises TypeError: Wrong type of at least one parameter
        :raises ValueError: Duplicate channels for at least one measure
        """
        super(Connector, self).__init__()
        if parent is not None and not isinstance(parent, BaseComponent):
            raise TypeError('Wrong type for parameter parent ({} != {})'.format(type(parent), BaseComponent))
        if not isinstance(channels, list):
            raise TypeError('Wrong type for parameter channels ({} != {})'.format(type(channels), list))
        for channel in channels:
            if not isinstance(channel, BaseChannel):
                # noinspection PyPep8
                raise TypeError('Wrong type for element in parameter channels ({} != {})'.format(type(channel), BaseChannel))
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
        # Set the internal states
        self._parent: Optional[BaseComponent] = parent
        self._channels: List[BaseChannel] = channels
        # Automatic register in component if type=Component
        from ..components.component import Component
        if parent is not None and isinstance(parent, Component):
            # noinspection PyProtectedMember
            parent._add_connector(self)  # pylint: disable=protected-access

    def _get_channels(self) -> List[BaseChannel]:
        """
        Internal method to return a list of included channels

        :return: List of included channels
        """
        return self._channels

    def _get_components(self) -> List[BaseComponent]:  # pylint: disable=no-self-use
        """
        Internal method to return a list of assigned components

        :return: List of assigned components
        """
        return [self._parent] if self._parent is not None else []
