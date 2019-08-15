#! /opt/conda/bin/python3
""" ProgressBar text output class """

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


import sys
import time


class ProgressBar:  # pylint: disable=too-many-instance-attributes
    """
    Class for displaying text style progress bar
    """

    MODE_SECONDS = 1
    MODE_MINUTES = 2
    MODE_HOURS = 3
    MODE_CLOCK = 4
    MODE_CLOCK_SHORT = 5
    MODE_CLOCK_LONG = 6

    def __init__(
            self
    ) -> None:
        """
        Initialization method
        """
        self._start_time = time.time()
        self._progress = 0.0
        self._number_format = '{:6.2f}%'
        self._bar_length = 30
        self._elapsed = True
        self._remain = True
        self._time_format = self.MODE_CLOCK
        self._status_text = ''
        self._clear_status_length = 0

    def reset(self) -> None:
        """
        Reset the time counter
        """
        self._start_time = time.time()
        self._progress = 0.0

    @property
    def progress(self) -> float:
        """
        Property: progress from 0 to 1

        :return: Current progress value
        """
        return self._progress

    @progress.setter
    def progress(self, value: float = None) -> None:
        if isinstance(value, (float, int)):
            self._progress = max(0.0, float(min(1.0, float(value))))

    @property
    def status(self) -> str:
        """
        Property: Status text to be displayed

        :return: Current status text
        """
        return self._status_text

    @status.setter
    def status(self, value: str) -> None:
        if not isinstance(value, str):
            value = ''
        new_clear = len(self._status_text) - (0 if value is None else len(value))
        self._clear_status_length = int(max(self._clear_status_length, new_clear))
        self._status_text = value

    def _format_seconds(self, elapsed: float) -> str:  # pylint: disable=too-many-return-statements
        """
        Format the given seconds according to internal style

        :param elapsed: Seconds of time
        :return: Formatted string
        """
        seconds = int(elapsed)
        # Mode seconds
        if self._time_format == self.MODE_SECONDS:
            return '{:5d}s'.format(seconds)
        # Mode minutes
        if self._time_format == self.MODE_MINUTES:
            return '{:4d}m'.format(int(round(seconds / 60)))
        # Mode hours
        if self._time_format == self.MODE_HOURS:
            return '{:3d}h'.format(int(round(seconds / 3600)))
        # Mode clock (short)
        minutes, seconds = divmod(seconds, 60)
        if self._time_format == self.MODE_CLOCK_SHORT:
            return '{:3d}:{:02d}'.format(minutes, seconds)
        # Mode clock
        hours, minutes = divmod(minutes, 60)
        if self._time_format == self.MODE_CLOCK:
            return '{:2d}:{:02d}:{:02d}'.format(hours, minutes, seconds)
        # Mode clock (long)
        days, hours = divmod(hours, 24)
        if self._time_format == self.MODE_CLOCK_LONG:
            return '{:d}d {:02d}:{:02d}:{:02d}'.format(days, hours, minutes, seconds)
        return '{:d}'.format(int(elapsed))

    def update(self, progress: float = None, status: str = None) -> None:
        """
        Update the text progress bar

        :param progress: New progress value from 0 to 1
        :param status: Status text to be shown
        """

        # Update values
        if isinstance(progress, float):
            self.progress = progress
        if isinstance(status, str):
            self.status = status

        # Calculate the times
        elapsed = time.time() - self._start_time
        remain = (elapsed / self.progress) * (1.0 - self.progress) if self.progress > 0.0 else 0.0

        # Initialize the printable text
        text = ''

        # Append the progress bar to the text
        add_text = self._number_format.format(100.0 * self.progress) if self._number_format is not None else ''
        if self._bar_length > 0:
            add_text = '[{{:<{}}}|{}]'\
                .format(self._bar_length, add_text)\
                .format("=" * int(self._bar_length * self.progress))
        text += add_text

        # Append the times
        if self._elapsed is True:
            text = self._format_seconds(elapsed) + ' ' + text
        if self._remain is True:
            text += ' ' + self._format_seconds(remain)

        # Append the status text
        if self._status_text != '' or self._clear_status_length > 0:
            text += '  '
        if self._status_text != '':
            text += self._status_text
        if self._clear_status_length > 0:
            text += '{{:<{:d}}}'.format(self._clear_status_length).format(' ' * self._clear_status_length)
            self._clear_status_length = 0

        # Print the final text
        sys.stdout.write("\r" + text)
        sys.stdout.flush()
