from unittest import TestCase
from cavsim.base.connectors.virtual_connector import VirtualConnector
from cavsim.base.connectors.connector import Connector
from cavsim.base.components.base_component import BaseComponent
from cavsim.base.channels.import_channel import ImportChannel
from cavsim.base.channels.export_channel import ExportChannel
from cavsim.measure import Measure


class TestVirtualConnector(TestCase):

    def test___init__(self):
        # Invalid parameter tests
        with self.assertRaises(TypeError):
            VirtualConnector('abc')
        with self.assertRaises(TypeError):
            VirtualConnector([Connector(None, []), 'abc'])
        # Valid parameter tests
        c = Connector(None, [])
        c._delegate = c
        with self.assertRaises(AssertionError):
            VirtualConnector([c])
        c = Connector(None, [ImportChannel(Measure.pressureLast)])
        c2 = Connector(None, [ImportChannel(Measure.pressureLast, True)])
        with self.assertRaises(TypeError):
            VirtualConnector([c, c2])
        c = Connector(None, [ExportChannel(Measure.pressureLast, lambda: 0)])
        c2 = Connector(None, [ExportChannel(Measure.pressureLast, lambda: 0)])
        with self.assertRaises(TypeError):
            VirtualConnector([c, c2])
        c = Connector(None, [ImportChannel(Measure.pressureLast)])
        c2 = Connector(None, [ExportChannel(Measure.pressureLast, lambda: 0)])
        vc = VirtualConnector([c, c2])
        self.assertCountEqual([c, c2], vc._connectors)
        self.assertEqual(vc, c._delegate)
        self.assertEqual(vc, c2._delegate)

    def test__get_channels(self):
        vc = VirtualConnector([])
        self.assertEqual([], vc._get_channels())
        c = Connector(None, [])
        vc._connectors = [c]
        self.assertEqual([], vc._get_channels())
        ic = ImportChannel(Measure.pressureCurrent)
        c = Connector(None, [ic])
        vc._connectors = [c]
        self.assertCountEqual([ic], vc._get_channels())
        ic2 = ImportChannel(Measure.pressureLast)
        c2 = Connector(None, [ic2])
        vc._connectors = [c, c2]
        self.assertCountEqual([ic, ic2], vc._get_channels())
        vc2 = VirtualConnector([])
        vc2._connectors = [vc]
        self.assertCountEqual([ic, ic2], vc._get_channels())

    def test__get_components(self):
        vc = VirtualConnector([])
        self.assertEqual([], vc._get_components())
        c = Connector(None, [])
        vc._connectors = [c]
        self.assertEqual([], vc._get_components())
        b = BaseComponent()
        c = Connector(b, [])
        vc._connectors = [c]
        self.assertCountEqual([b], vc._get_components())
        b2 = BaseComponent()
        c2 = Connector(b2, [])
        vc._connectors = [c, c2]
        self.assertCountEqual([b, b2], vc._get_components())
        vc2 = VirtualConnector([])
        vc2._connectors = [vc]
        self.assertCountEqual([b, b2], vc._get_components())

    def test_release(self):
        c = Connector(None, [])
        c2 = Connector(None, [])
        vc = VirtualConnector([])
        c._delegate = vc
        c2._delegate = vc
        vc._connectors = [c, c2]
        vc.release()
        self.assertEqual(None, c._delegate)
        self.assertEqual(None, c2._delegate)
        self.assertEqual([], vc._connectors)
