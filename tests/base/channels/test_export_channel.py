from unittest import TestCase
from cavsim.base.channels.export_channel import ExportChannel
from cavsim.measure import Measure


class TestExportChannel(TestCase):

    def _dummy(self):
        return 123

    def test___init__(self):
        # Invalid tests
        with self.assertRaises(TypeError):
            ExportChannel(123, lambda: None)
        # Valid tests
        c = ExportChannel(Measure.pressureLast, self._dummy)
        self.assertEqual(Measure.pressureLast, c.measure)
        self.assertEqual(False, c.is_import)
        self.assertEqual(self._dummy, c._callback)
        with self.assertRaises(TypeError):
            ExportChannel(Measure.pressureLast, 123)

    def test_export_value(self):
        c = ExportChannel(Measure.pressureLast, self._dummy)
        self.assertEqual(123, c.export_value())
        c = ExportChannel(Measure.pressureLast, lambda: 'abc')
        self.assertEqual('abc', c.export_value())
