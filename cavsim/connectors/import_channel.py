#! /opt/conda/bin/python3
""" Class for connectors import data channels """

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


from typing import Any
from ..measure import Measure
from .channel import Channel
from .export_channel import ExportChannel


class ImportChannel(Channel):
    """
    Importing channel class for transferring measures between two connected channels
    """

    def __init__(self, measure: Measure, optional: bool = False, default: Any = None) -> None:
        """
        Initialization of the import channel

        :param measure: Measure unit being transferred
        :param optional: Whether the import is optional
        :param default: Default value for optional imports when not connected
        :raises TypeError: Wrong type of at least one parameter
        """
        super(ImportChannel, self).__init__(measure, True, optional)
        if not isinstance(measure, Measure):
            raise TypeError('Wrong type for parameter measure ({} != {})'.format(type(measure), Measure))
        if not isinstance(optional, bool):
            raise TypeError('Wrong type for parameter optional ({} != {})'.format(type(optional), bool))
        self._default: Any = default

    @property
    def default(self) -> Any:
        """
        Default value for optional imports

        :return: Default value for optional imports
        """
        return self._default

    def import_value(self) -> Any:
        """
        Import the value for a connected export channel

        :return: Imported value
        :raises AssertionError: Channel not connected
        :raises TypeError: Channel not connected to an export channel
        """
        if self.connected is False:
            if self.optional is False:
                raise AssertionError('Channel must be connected to import a value!')
            return self._default
        if not isinstance(self._connection, ExportChannel):
            raise TypeError('Type ({}) of connected channel is no ExportChannel!'.format(type(self._connection)))
        return self._connection.export_value()
