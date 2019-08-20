from unittest import TestCase
from cavsim.base.connectors.connector import Connector
from cavsim.base.components.base_component import BaseComponent
from cavsim.base.components.component import Component
from cavsim.base.channels.base_channel import BaseChannel
from cavsim.base.channels.import_channel import ImportChannel
from cavsim.base.channels.export_channel import ExportChannel
from cavsim.measure import Measure


class TestConnector(TestCase):

    def test___init__(self):
        # Test invalid parameters
        bc = BaseComponent()
        ic = ImportChannel(Measure.pressureCurrent)
        with self.assertRaises(TypeError):
            c = Connector(123, [])
        with self.assertRaises(TypeError):
            c = Connector(bc, 123)
        with self.assertRaises(TypeError):
            c = Connector(bc, [BaseChannel(Measure.pressureCurrent, True), 'abc'])
        # Valid parameter tests
        with self.assertRaises(ValueError):
            c = Connector(bc, [ImportChannel(Measure.pressureCurrent), ImportChannel(Measure.pressureCurrent)])
        with self.assertRaises(ValueError):
            c = Connector(bc, [ImportChannel(Measure.pressureCurrent), ImportChannel(Measure.pressureCurrent, True)])
        with self.assertRaises(ValueError):
            c = Connector(bc, [ImportChannel(Measure.pressureCurrent, True), ImportChannel(Measure.pressureCurrent, True)])
        with self.assertRaises(ValueError):
            c = Connector(bc, [ExportChannel(Measure.pressureCurrent, lambda: 0), ExportChannel(Measure.pressureCurrent, lambda: 0)])
        Connector(bc, [ImportChannel(Measure.pressureCurrent), ImportChannel(Measure.pressureLast, True), ExportChannel(Measure.pressureCurrent, lambda: 0)])
        c = Connector(bc, [ic])
        self.assertEqual(c._parent, bc)
        self.assertCountEqual(c._channels, [ic])
        comp = Component()
        c = Connector(comp, [ic])
        self.assertCountEqual(comp._connectors, [c])

    def test__get_channels(self):
        c = Connector(None, [])
        c._channels = None
        self.assertEqual(c._get_channels(), None)
        c._channels = c
        self.assertEqual(c._get_channels(), c)
        c._channels = 'abc'
        self.assertEqual(c._get_channels(), 'abc')

    def test__get_components(self):
        c = Connector(None, [])
        c._parent = None
        self.assertEqual(c._get_components(), [])
        c._parent = c
        self.assertCountEqual(c._get_components(), [c])
        c._parent = 'abc'
        self.assertCountEqual(c._get_components(), ['abc'])
