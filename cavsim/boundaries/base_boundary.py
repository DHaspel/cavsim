#! /opt/conda/bin/python3
""" Base pipe class containing shared boundary properties """

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


from typing import Callable
import numpy as np
from ..base.components.numerical_component import NumericalComponent


BoundaryFunction = Callable[[float], float]


class BaseBoundary(NumericalComponent):
    """
    Basic boundary class with shared properties for all boundaries
    """

    def __init__(self) -> None:
        """
        Initialization of the class
        """
        super(BaseBoundary, self).__init__()
