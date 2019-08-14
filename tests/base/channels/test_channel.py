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
        self.assertEqual(c._measure, Measure.pressureLast)
        self.assertEqual(c._is_import, False)
        self.assertEqual(c._connection, None)
        self.assertEqual(c._optional, False)
        c = Channel(Measure.pressureCurrent, True, True)
        self.assertEqual(c._measure, Measure.pressureCurrent)
        self.assertEqual(c._is_import, True)
        self.assertEqual(c._connection, None)
        self.assertEqual(c._optional, True)

    def test___del__(self):
        c = Channel(Measure.pressureLast, False)
        c2 = Channel(Measure.pressureLast, False)
        c._connection = c2
        c2._connection = c
        c.__del__()
        self.assertEqual(c._connection, None)
        self.assertEqual(c2._connection, None)

    def test_measure(self):
        c = Channel(Measure.pressureLast, False)
        self.assertEqual(c.measure, Measure.pressureLast)
        c._measure = Measure.velocityPlusLast
        self.assertEqual(c.measure, Measure.velocityPlusLast)
        with self.assertRaises(AttributeError):
            c.measure = Measure.pressureCurrent

    def test_connected(self):
        c = Channel(Measure.pressureLast, False)
        self.assertEqual(c.connected, False)
        c._connection = c
        self.assertEqual(c.connected, True)
        with self.assertRaises(AttributeError):
            c.connected = False

    def test_optional(self):
        c = Channel(Measure.pressureLast, False)
        self.assertEqual(c.optional, False)
        c._optional = True
        self.assertEqual(c.optional, True)
        with self.assertRaises(AttributeError):
            c.optional = False

    def test_is_import(self):
        c = Channel(Measure.pressureLast, False)
        self.assertEqual(c.is_import, False)
        c._is_import = True
        self.assertEqual(c.is_import, True)
        with self.assertRaises(AttributeError):
            c.is_import = False

    def test_is_export(self):
        c = Channel(Measure.pressureLast, False)
        self.assertEqual(c.is_export, True)
        c._is_import = True
        self.assertEqual(c.is_export, False)
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
        self.assertEqual(c.connectable(c2), False)
        c = Channel(Measure.pressureLast, False)
        c2 = Channel(Measure.pressureLast, False)
        self.assertEqual(c.connectable(c2), False)
        c = Channel(Measure.pressureLast, False)
        c2 = Channel(Measure.pressureLast, True)
        self.assertEqual(c.connectable(c2), True)
        c = Channel(Measure.pressureLast, False)
        c2 = Channel(Measure.pressureLast, True)
        c._connection = c2
        self.assertEqual(c.connectable(c2), False)
        c._connection = None
        c2._connection = c
        self.assertEqual(c.connectable(c2), False)
        c._connection = c2
        self.assertEqual(c.connectable(c2), False)

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
        self.assertEqual(c._connection, c2)
        self.assertEqual(c2._connection, c)

    def test_disconnect(self):
        c = Channel(Measure.pressureLast, False)
        c2 = Channel(Measure.pressureLast, False)
        c._connection = c2
        c2._connection = c
        c.disconnect()
        self.assertEqual(c._connection, None)
        self.assertEqual(c2._connection, None)
