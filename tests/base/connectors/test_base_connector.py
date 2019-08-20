from unittest import TestCase
from cavsim.base.connectors.base_connector import BaseConnector
from cavsim.base.channels.base_channel import BaseChannel as Channel
from cavsim.base.channels.import_channel import ImportChannel
from cavsim.base.channels.export_channel import ExportChannel
from cavsim.measure import Measure


class DelegateError(BaseException):
    pass


class DummyConnector():
    def __call__(self, *args, **kwargs):
        raise DelegateError('Delegation!')
    def __getattr__(self, item):
        raise DelegateError('Delegation!')


class WrapperBaseConnector(BaseConnector):
    def __init__(self):
        super(WrapperBaseConnector, self).__init__()
        self._channels = [
            ImportChannel(Measure.pressureCurrent),
            ImportChannel(Measure.pressureLast),
            ImportChannel(Measure.deltaX, True),
            ExportChannel(Measure.diameter, lambda: 0)
        ]
    def _get_channels(self):
        return self._channels


class WrapperGetChannels(BaseConnector):
    def _get_channels(self):
        return [1,2,3]
    def _get_components(self):
        return [4,5,6]


class TestBaseConnector(TestCase):

    def test___init__(self):
        c = BaseConnector()
        self.assertEqual(c._delegate, None)
        self.assertEqual(c._link, None)

    def test_delegate(self):
        c = BaseConnector()
        d = DummyConnector()
        self.assertEqual(c._delegate, None)
        c._delegate = d
        with self.assertRaises(DelegateError):
            c.link
        with self.assertRaises(DelegateError):
            c.connected
        with self.assertRaises(DelegateError):
            c.channels
        with self.assertRaises(DelegateError):
            c.components
        with self.assertRaises(DelegateError):
            c.imports
        with self.assertRaises(DelegateError):
            c.exports
        with self.assertRaises(DelegateError):
            c.optionals
        with self.assertRaises(DelegateError):
            c.connectable(c)
        with self.assertRaises(DelegateError):
            c.disconnect()
        with self.assertRaises(DelegateError):
            c.connect(c)
        with self.assertRaises(DelegateError):
            c.value(Measure.pressureLast)

    def test_link(self):
        c = BaseConnector()
        self.assertEqual(c.link, None)
        c._link = c
        self.assertEqual(c.link, c)

    def test_connected(self):
        c = WrapperBaseConnector()
        self.assertEqual(c.connected, False)
        c._get_channels()[0].connect(ExportChannel(Measure.pressureCurrent, lambda: 0))
        with self.assertRaises(ValueError):
            c.connected
        c._get_channels()[1].connect(ExportChannel(Measure.pressureLast, lambda: 0))
        self.assertEqual(c.connected, True)
        c = BaseConnector()
        c._link = c
        self.assertEqual(c.connected, True)
        c = WrapperBaseConnector()
        c._link = c
        with self.assertRaises(ValueError):
            c.connected

    def test_channels(self):
        c = BaseConnector()
        self.assertEqual(c.channels, [])
        c = WrapperGetChannels()
        self.assertCountEqual(c.channels, [1,2,3])

    def test__get_channels(self):
        c = BaseConnector()
        self.assertEqual(c._get_channels(), [])

    def test_components(self):
        c = BaseConnector()
        self.assertEqual(c.components, [])
        c = WrapperGetChannels()
        self.assertCountEqual(c.components, [4,5,6])

    def test__get_components(self):
        c = BaseConnector()
        self.assertEqual(c._get_components(), [])

    def test_imports(self):
        c = BaseConnector()
        self.assertEqual(c.imports, set())
        c = WrapperBaseConnector()
        self.assertEqual(c.imports, {Measure.pressureCurrent, Measure.pressureLast})

    def test_optionals(self):
        c = BaseConnector()
        self.assertEqual(c.optionals, set())
        c = WrapperBaseConnector()
        self.assertEqual(c.optionals, {Measure.deltaX})

    def test_exports(self):
        c = BaseConnector()
        self.assertEqual(c.exports, set())
        c = WrapperBaseConnector()
        self.assertEqual(c.exports, {Measure.diameter})

    def test_connectable(self):
        c = BaseConnector()
        c2 = BaseConnector()
        with self.assertRaises(TypeError):
            c.connectable(123)
        self.assertEqual(c.connectable(c2), True)
        c = WrapperBaseConnector()
        c2 = WrapperBaseConnector()
        self.assertEqual(c.connectable(c2), False)
        c._channels.append(ExportChannel(Measure.pressureCurrent, lambda: 0))
        c._channels.append(ExportChannel(Measure.pressureLast, lambda: 0))
        self.assertEqual(c.connectable(c2), False)
        c2._channels.append(ExportChannel(Measure.pressureCurrent, lambda: 0))
        c2._channels.append(ExportChannel(Measure.pressureLast, lambda: 0))
        self.assertEqual(c.connectable(c2), True)
        c._get_channels()[0].connect(ExportChannel(Measure.pressureCurrent, lambda: 0))
        c._get_channels()[1].connect(ExportChannel(Measure.pressureLast, lambda: 0))
        self.assertEqual(c.connectable(c2), False)

    def test_disconnect(self):
        c = WrapperBaseConnector()
        c2 = BaseConnector()
        c._get_channels()[0].connect(ExportChannel(Measure.pressureCurrent, lambda: 0))
        c._get_channels()[1].connect(ExportChannel(Measure.pressureLast, lambda: 0))
        c._link = c2
        c2._link = c
        self.assertEqual(c._get_channels()[0].connected, True)
        self.assertEqual(c._get_channels()[1].connected, True)
        c.disconnect()
        self.assertEqual(c._link, None)
        self.assertEqual(c2._link, None)
        self.assertEqual(c._get_channels()[0].connected, False)
        self.assertEqual(c._get_channels()[1].connected, False)

    def test_connect(self):
        # Invalid parameter tests
        c = BaseConnector()
        with self.assertRaises(TypeError):
            c.connect(123)
        c2 = BaseConnector()
        c2._delegate = c2
        with self.assertRaises(AssertionError):
            c.connect(c2)
        c2._delegate = None
        c2._link = c
        with self.assertRaises(AssertionError):
            c.connect(c2)
        c = WrapperBaseConnector()
        c2 = WrapperBaseConnector()
        with self.assertRaises(TypeError):
            c.connect(c2)
        # Valid type tests
        c = BaseConnector()
        c2 = BaseConnector()
        c.connect(c2)
        self.assertEqual(c.connected, True)
        self.assertEqual(c._link, c2)
        self.assertEqual(c2._link, c)
        # Test channel linking
        c = WrapperBaseConnector()
        c._channels.append(ExportChannel(Measure.pressureCurrent, lambda: 99))
        c._channels.append(ExportChannel(Measure.pressureLast, lambda: 88))
        c2 = WrapperBaseConnector()
        c2._channels.append(ExportChannel(Measure.pressureCurrent, lambda: 77))
        c2._channels.append(ExportChannel(Measure.pressureLast, lambda: 66))
        c2._channels.append(ExportChannel(Measure.deltaX, lambda: 55))
        c.connect(c2)
        self.assertEqual(c._channels[0].import_value(), 77)
        self.assertEqual(c._channels[1].import_value(), 66)
        self.assertEqual(c._channels[2].import_value(), 55)
        self.assertEqual(c2._channels[0].import_value(), 99)
        self.assertEqual(c2._channels[1].import_value(), 88)
        self.assertEqual(c2._channels[2].import_value(), None)

    def test_value(self):
        c = WrapperBaseConnector()
        with self.assertRaises(AssertionError):
            c.value(Measure.pressureCurrent)
        c._get_channels()[0].connect(ExportChannel(Measure.pressureCurrent, lambda: 56))
        c._get_channels()[1].connect(ExportChannel(Measure.pressureLast, lambda: 78))
        self.assertEqual(c.value(Measure.pressureCurrent), 56)
        self.assertEqual(c.value(Measure.pressureLast), 78)
        with self.assertRaises(ValueError):
            c.value(Measure.diameter)
