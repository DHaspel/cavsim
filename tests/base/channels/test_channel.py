from unittest import TestCase
from cavsim.base.channels.base_channel import BaseChannel as Channel
from cavsim.measure import Measure


class TestChannel(TestCase):

    def test___init__(self):
        # Invalid tests
        with self.assertRaises(TypeError):
            c = Channel(123, False)
        with self.assertRaises(TypeError):
            c = Channel(Measure.pressureLast, 'abc')
        with self.assertRaises(TypeError):
            c = Channel(Measure.pressureLast, False, 'abc')
        # Valid tests
        c = Channel(Measure.pressureLast, False)
        self.assertEqual(Measure.pressureLast, c._measure)
        self.assertEqual(False, c._is_import)
        self.assertEqual(None, c._connection)
        self.assertEqual(False, c._optional)
        c = Channel(Measure.pressureCurrent, True, True)
        self.assertEqual(Measure.pressureCurrent, c._measure)
        self.assertEqual(True, c._is_import)
        self.assertEqual(None, c._connection)
        self.assertEqual(True, c._optional)

    def test_measure(self):
        c = Channel(Measure.pressureLast, False)
        self.assertEqual(Measure.pressureLast, c.measure)
        c._measure = Measure.velocityPlusLast
        self.assertEqual(Measure.velocityPlusLast, c.measure)
        with self.assertRaises(AttributeError):
            c.measure = Measure.pressureCurrent

    def test_connected(self):
        c = Channel(Measure.pressureLast, False)
        self.assertEqual(False, c.connected)
        c._connection = c
        self.assertEqual(True, c.connected)
        with self.assertRaises(AttributeError):
            c.connected = False

    def test_optional(self):
        c = Channel(Measure.pressureLast, False)
        self.assertEqual(False, c.optional)
        c._optional = True
        self.assertEqual(True, c.optional)
        with self.assertRaises(AttributeError):
            c.optional = False

    def test_is_import(self):
        c = Channel(Measure.pressureLast, False)
        self.assertEqual(False, c.is_import)
        c._is_import = True
        self.assertEqual(True, c.is_import)
        with self.assertRaises(AttributeError):
            c.is_import = False

    def test_is_export(self):
        c = Channel(Measure.pressureLast, False)
        self.assertEqual(True, c.is_export)
        c._is_import = True
        self.assertEqual(False, c.is_export)
        with self.assertRaises(AttributeError):
            c.is_export = False

    def test_connectable(self):
        # Invalid tests
        c = Channel(Measure.pressureLast, False)
        with self.assertRaises(TypeError):
            c.connectable(123)
        # Valid tests
        c = Channel(Measure.pressureLast, False)
        c2 = Channel(Measure.pressureCurrent, True)
        self.assertEqual(False, c.connectable(c2))
        c = Channel(Measure.pressureLast, False)
        c2 = Channel(Measure.pressureLast, False)
        self.assertEqual(False, c.connectable(c2))
        c = Channel(Measure.pressureLast, False)
        c2 = Channel(Measure.pressureLast, True)
        self.assertEqual(True, c.connectable(c2))
        c = Channel(Measure.pressureLast, False)
        c2 = Channel(Measure.pressureLast, True)
        c._connection = c2
        self.assertEqual(False, c.connectable(c2))
        c._connection = None
        c2._connection = c
        self.assertEqual(False, c.connectable(c2))
        c._connection = c2
        self.assertEqual(False, c.connectable(c2))

    def test_connect(self):
        # Invalid tests
        c = Channel(Measure.pressureLast, False)
        with self.assertRaises(TypeError):
            c.connect(123)
        # Valid tests
        with self.assertRaises(TypeError):
            c = Channel(Measure.pressureLast, False)
            c2 = Channel(Measure.pressureCurrent, True)
            c.connect(c2)
        with self.assertRaises(TypeError):
            c = Channel(Measure.pressureLast, False)
            c2 = Channel(Measure.pressureLast, False)
            c.connect(c2)
        c = Channel(Measure.pressureLast, False)
        c2 = Channel(Measure.pressureLast, True)
        c._connection = c2
        with self.assertRaises(AssertionError):
            c.connect(c2)
        c._connection = None
        c2._connection = c
        with self.assertRaises(AssertionError):
            c.connect(c2)
        c._connection = c2
        with self.assertRaises(AssertionError):
            c.connect(c2)
        c = Channel(Measure.pressureLast, False)
        c2 = Channel(Measure.pressureLast, True)
        c.connect(c2)
        self.assertEqual(c2, c._connection)
        self.assertEqual(c, c2._connection)

    def test_disconnect(self):
        c = Channel(Measure.pressureLast, False)
        c2 = Channel(Measure.pressureLast, False)
        c._connection = c2
        c2._connection = c
        c.disconnect()
        self.assertEqual(None, c._connection)
        self.assertEqual(None, c2._connection)
