#! /opt/conda/bin/python3
""" File defining CavSims subpackage containing the boundary classes """

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


from .base_boundary import BaseBoundary
from .left_boundary_pressure import LeftBoundaryPressure
from .left_boundary_velocity import LeftBoundaryVelocity
from .right_boundary_pressure import RightBoundaryPressure
from .right_boundary_velocity import RightBoundaryVelocity
