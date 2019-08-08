from unittest import TestCase
from cavsim.connectors.channel import Channel
from cavsim.connectors.import_channel import ImportChannel
from cavsim.connectors.export_channel import ExportChannel
from cavsim.measure import Measure


class TestExportChannel(TestCase):

    def _dummy(self):
        return 123

    def test___init__(self):
        c = ExportChannel(Measure.pressureLast, self._dummy)
        self.assertEqual(c.measure, Measure.pressureLast)
        self.assertEqual(c.is_import, False)
        self.assertEqual(c._callback, self._dummy)
        with self.assertRaises(TypeError):
            c = ExportChannel(Measure.pressureLast, 123)

    def test_export_value(self):
        c = ExportChannel(Measure.pressureLast, self._dummy)
        self.assertEqual(c.export_value(), 123)
        c = ExportChannel(Measure.pressureLast, lambda: 'abc')
        self.assertEqual(c.export_value(), 'abc')
