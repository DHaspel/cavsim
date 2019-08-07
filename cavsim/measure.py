#! /opt/conda/bin/python3
""" EnumClass defining measures which are used in the simulation """

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


from enum import Enum, unique, auto


@unique
class Measure(Enum):
    """
    Enumeration class for measures used in the simulation
    """
    pressureCurrent = auto()
    pressureLast = auto()
    velocityCurrent = auto()
    velocityLast = auto()
    volumeFlowCurrent = auto()
    volumeFlowLast = auto()
