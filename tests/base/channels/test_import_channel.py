from unittest import TestCase
from cavsim.base.channels.base_channel import BaseChannel as Channel
from cavsim.base.channels.import_channel import ImportChannel
from cavsim.base.channels.export_channel import ExportChannel
from cavsim.measure import Measure


class TestImportChannel(TestCase):

    def test___init__(self):
        # Invalid tests
        with self.assertRaises(TypeError):
            ImportChannel(123)
        with self.assertRaises(TypeError):
            ImportChannel(Measure.pressureLast, 123)
        # Valid tests
        c = ImportChannel(Measure.pressureLast)
        self.assertEqual(Measure.pressureLast, c.measure)
        self.assertEqual(True, c.is_import)
        self.assertEqual(False, c.optional)
        self.assertEqual(None, c._default)
        c = ImportChannel(Measure.pressureLast, True, 123)
        self.assertEqual(True, c.optional)
        self.assertEqual(123, c._default)

    def test_default(self):
        c = ImportChannel(Measure.pressureLast)
        self.assertEqual(None, c.default)
        c._default = 456
        self.assertEqual(456, c.default)

    def test_import_value(self):
        c = ImportChannel(Measure.pressureLast)
        with self.assertRaises(AssertionError):
            c.import_value()
        c = ImportChannel(Measure.pressureLast, True, 789)
        self.assertEqual(789, c.import_value())
        c2 = Channel(Measure.pressureLast, False)
        c.connect(c2)
        with self.assertRaises(TypeError):
            c.import_value()
        c.disconnect()
        c2 = ExportChannel(Measure.pressureLast, lambda: 159)
        c.connect(c2)
        self.assertEqual(159, c.import_value())

