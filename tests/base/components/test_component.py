from unittest import TestCase
from cavsim.base.components.component import Component
from cavsim.base.fluids.base_fluid import BaseFluid
from cavsim.base.connectors.base_connector import BaseConnector
from cavsim.base.connectors.connector import Connector


class Connected(BaseConnector):
    @property
    def connected(self):
        return True


class Unconnected(BaseConnector):
    @property
    def connected(self):
        return False


class ToDisconnect(BaseConnector):
    def __init__(self):
        self._called = False
    def disconnect(self):
        self._called = True


class TestComponent(TestCase):

    def test___init__(self):
        c = Component()
        self.assertEqual([], c._connectors)

    def test_connectors(self):
        c = Component()
        self.assertEqual([], c.connectors)
        c._connectors = 123
        self.assertEqual(123, c.connectors)
        c._connectors = 'abc'
        self.assertEqual('abc', c.connectors)

    def test_connector_count(self):
        c = Component()
        c._connectors = [1,2,3]
        self.assertEqual(3, c.connector_count)
        c._connectors = [1,2,3,4,5]
        self.assertEqual(5, c.connector_count)
        c._connectors = []
        self.assertEqual(0, c.connector_count)

    def test_connected(self):
        c = Component()
        self.assertEqual(0, c.connected)
        c._connectors = [Connected(), Connected(), Unconnected()]
        self.assertEqual(2, c.connected)

    def test_unconnected(self):
        c = Component()
        self.assertEqual(0, c.unconnected)
        c._connectors = [Connected(), Connected(), Unconnected()]
        self.assertEqual(1, c.unconnected)

    def test_disconnect(self):
        c = Component()
        c.disconnect()
        c1 = ToDisconnect()
        c2 = ToDisconnect()
        c._connectors = [c1, c2]
        self.assertEqual(False, c1._called)
        self.assertEqual(False, c2._called)
        c.disconnect()
        self.assertEqual(True, c1._called)
        self.assertEqual(True, c2._called)

    def test_connect(self):
        # Test invalid types
        c = Component()
        with self.assertRaises(TypeError):
            c.connect(123)
        # Test no matches
        c1 = Component()
        c2 = Component()
        with self.assertRaises(AssertionError):
            c1.connect(c2)
        # Test too many matches
        c1 = Component()
        c1._connectors = [Connector(None,[]), Connector(None,[])]
        c2 = Component()
        c2._connectors = [Connector(None,[]), Connector(None,[])]
        with self.assertRaises(AssertionError):
            c1.connect(c2)
        with self.assertRaises(AssertionError):
            c1.connect(c2._connectors[0])
        # Test valid matches
        c1 = Component()
        c1._connectors = [Connector(None,[])]
        c2 = Component()
        c2._connectors = [Connector(None,[]), Connector(None,[])]
        with self.assertRaises(AssertionError):
            c1.connect(c2)
        c1.connect(c2._connectors[0])
        c1 = Component()
        c1._connectors = [Connector(None,[])]
        c2 = Component()
        c2._connectors = [Connector(None,[])]
        c1.connect(c2)

    def test__add_connector(self):
        c = Component()
        with self.assertRaises(TypeError):
            c._add_connector(123)
        c1 = BaseConnector()
        c2 = BaseConnector()
        c._connectors = [c1,c2]
        c._add_connector(c1)
        self.assertCountEqual([c1,c2], c._connectors)
        c._connectors = [c1]
        c._add_connector(c1)
        self.assertCountEqual([c1], c._connectors)
        c._connectors = [c1]
        c._add_connector(c2)
        self.assertCountEqual([c1,c2], c._connectors)
