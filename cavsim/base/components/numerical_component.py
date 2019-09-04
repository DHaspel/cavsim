#! /opt/conda/bin/python3
""" NumericalComponent class to implement basic data field features """

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


from typing import Tuple, Dict
import numpy as np
from .component import Component


class NumericalComponent(Component):
    """
    NumericalComponent class to implement basic data field methods
    """

    def __init__(self) -> None:
        """
        Initialization of the class
        """
        super(NumericalComponent, self).__init__()
        self._fields: Dict[str, Tuple[int, np.ndarray]] = {}
        self._delta_t = 0.0
        self._delta_x = 0.0

    @property
    def fields(self) -> Dict[str, Tuple[int, np.ndarray]]:
        """
        Fields property (registered fields)

        :return: Dictionary of tuples (time_dim, numpy array) for registered fields
        """
        return self._fields

    def field(self, name: str) -> np.ndarray:
        """
        Get a specific fields numpy array

        :param name: Name of the field to get numpy array for
        :return: Numpy array of the field
        :raises TypeError: Wrong type of at least one parameter
        :raises KeyError: No field registered under the given name
        """
        if not isinstance(name, str):
            raise TypeError('Wrong type for parameter name ({} != {})'.format(type(name), str))
        if name not in self._fields.keys():
            raise KeyError('Field with key "{}" does not exists!'.format(name))
        return self._fields[name][1]

    def field_slice(self, name: str, time_offset: int = 0, x_offset: int = 0) -> np.ndarray:
        """
        Get a slice of the specified field

        The method returns a numpy array slice for the field specified by the name,
        which contains the values at a certain time offset and with the specified
        x offset. The shape of the returns array is always (x-dimension - 2,).

        :param name: Name of the field to get slice for
        :param time_offset: Time offset of the slice
        :param x_offset: X offset of the slice
        :return: Slice of the field specified by parameters
        :raises TypeError: Wrong type of at least one parameter
        :raises KeyError: No field registered under the given name
        :raises KeyError: Index of time or x offset out of bounds
        """
        if not isinstance(name, str):
            raise TypeError('Wrong type for parameter name ({} != {})'.format(type(name), str))
        if not isinstance(time_offset, int):
            raise TypeError('Wrong type for parameter time_offset ({} != {})'.format(type(time_offset), int))
        if not isinstance(x_offset, int):
            raise TypeError('Wrong type for parameter x_offset ({} != {})'.format(type(x_offset), int))
        if name not in self._fields.keys():
            raise KeyError('Field with key "{}" does not exists!'.format(name))
        if x_offset < -1 or x_offset > 1:
            raise KeyError('X-Offset ({}) of the slice needs to be between -1 and 1!'.format(x_offset))
        time_steps, field = self._fields[name]
        if time_offset < 0 or time_offset >= time_steps:
            # noinspection PyPep8
            raise KeyError('Time-Offset ({}) of the slice needs to be between 0 and {}!'.format(time_offset, time_steps-1))
        if x_offset == 1:
            return field[time_offset, 2:]
        return field[time_offset, (1 + x_offset):(-1 + x_offset)]

    def field_create(self, name: str, time_steps: int) -> np.ndarray:
        """
        Create and register a new field

        :param name: Name of the field to be registered
        :param time_steps: Number of time steps to create the field for
        :return: Numpy array of the newly created field
        :raises TypeError: Wrong type of at least one parameter
        :raises KeyError: Another field was already registered to this name
        """
        if not isinstance(name, str):
            raise TypeError('Wrong type for parameter name ({} != {})'.format(type(name), str))
        if not isinstance(time_steps, int):
            raise TypeError('Wrong type for parameter time_steps ({} != {})'.format(type(time_steps), int))
        if name in self._fields.keys():
            raise KeyError('Field with key "{}" already exists!'.format(name))
        field = np.zeros((time_steps, 0))
        self._fields[name] = (time_steps, field)
        return field

    def fields_resize(self, new_size: int) -> None:
        """
        Resize all registered fields to the given x-dimension

        :param new_size: New x-dimension to resize the fields to
        :raises TypeError: Wrong type of at least one parameter
        """
        if not isinstance(new_size, int):
            raise TypeError('Wrong type for parameter new_size ({} != {})'.format(type(new_size), int))
        for time_steps, field in self._fields.values():
            field.resize((time_steps, new_size), refcheck=False)

    def fields_move(self) -> None:
        """
        Move all entries in all registered fields one timestep into the past
        """
        for _, field in self._fields.values():
            field[1:, :] = field[:-1, :]

    def field_wide_slice(self, name: str, time_offset: int = 0) -> np.ndarray:
        """
        Get a wide slice of the specified field

        The method returns a numpy array slice for the field specified by the name,
        which contains the values at a certain time offset. The shape of the returns
        array is always (x-dimension,).

        :param name: Name of the field to get slice for
        :param time_offset: Time offset of the slice
        :return: Wide slice of the field specified by parameters
        :raises TypeError: Wrong type of at least one parameter
        :raises KeyError: No field registered under the given name
        :raises KeyError: Index of time or x offset out of bounds
        """
        # todo: Unit-testing + comment checking
        if not isinstance(name, str):
            raise TypeError('Wrong type for parameter name ({} != {})'.format(type(name), str))
        if not isinstance(time_offset, int):
            raise TypeError('Wrong type for parameter time_offset ({} != {})'.format(type(time_offset), int))
        if name not in self._fields.keys():
            raise KeyError('Field with key "{}" does not exists!'.format(name))
        time_steps, field = self._fields[name]
        if time_offset < 0 or time_offset >= time_steps:
            # noinspection PyPep8
            raise KeyError('Time-Offset ({}) of the slice needs to be between 0 and {}!'.format(time_offset, time_steps-1))
        return field[time_offset, :]
