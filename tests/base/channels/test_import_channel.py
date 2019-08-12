from unittest import TestCase
from cavsim.base.channels.base_channel import BaseChannel as Channel
from cavsim.base.channels.import_channel import ImportChannel
from cavsim.base.channels.export_channel import ExportChannel
from cavsim.measure import Measure


class TestImportChannel(TestCase):

    def test___init__(self):
        # Invalid tests
        with self.assertRaises(TypeError):
            c = ImportChannel(123)
        with self.assertRaises(TypeError):
            c = ImportChannel(Measure.pressureLast, 123)
        # Valid tests
        c = ImportChannel(Measure.pressureLast)
        self.assertEqual(c.measure, Measure.pressureLast)
        self.assertEqual(c.is_import, True)
        self.assertEqual(c.optional, False)
        self.assertEqual(c._default, None)
        c = ImportChannel(Measure.pressureLast, True, 123)
        self.assertEqual(c.optional, True)
        self.assertEqual(c._default, 123)

    def test_default(self):
        c = ImportChannel(Measure.pressureLast)
        self.assertEqual(c.default, None)
        c._default = 456
        self.assertEqual(c.default, 456)

    def test_import_value(self):
        c = ImportChannel(Measure.pressureLast)
        with self.assertRaises(AssertionError):
            c.import_value()
        c = ImportChannel(Measure.pressureLast, True, 789)
        self.assertEqual(c.import_value(), 789)
        c2 = Channel(Measure.pressureLast, False)
        c.connect(c2)
        with self.assertRaises(TypeError):
            c.import_value()
        c.disconnect()
        c2 = ExportChannel(Measure.pressureLast, lambda: 159)
        c.connect(c2)
        self.assertEqual(c.import_value(), 159)

