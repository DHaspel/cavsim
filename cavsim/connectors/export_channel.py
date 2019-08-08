#! /opt/conda/bin/python3
""" Class for connectors export data channels """

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


from typing import Any, Callable
from ..measure import Measure
from .channel import Channel


class ExportChannel(Channel):
    """
    Exporting channel class for transferring measures between two connected channels
    """

    def __init__(self, measure: Measure, callback: Callable[[], Any]) -> None:
        """
        Initialization of the export channel

        :param measure: Measure unit being transferred
        :param callback: Callback method to get the value being exported
        :raises TypeError: Callback not a function
        """
        super(ExportChannel, self).__init__(measure, False, False)
        if not callable(callback):
            raise TypeError('Callback needs to be a function (Callable[[],Any])!')
        self._callback: Callable[[], Any] = callback

    def export_value(self) -> Any:
        """
        Return the exported value

        :return: Exported value
        """
        return self._callback()
