from unittest import TestCase
from cavsim.base.solvers.base_solver import BaseSolver
from cavsim.base.components.component import Component
from cavsim.base.fluids.base_fluid import BaseFluid
from cavsim.base.connectors.connector import Connector


class DummySolver(BaseSolver):
    def __init__(self, result):
        self._result = result
        self._seeds = []
    def _get_connected_list(self, *args, **kwargs):
        if self._result is False:
            raise AssertionError('DUMMY CLASS ERROR!')


class DummySolver2(BaseSolver):
    def __init__(self):
        self._seeds = []
        self._call_seeds = None
        self._call_boolean = None
    def _get_connected_list(self, seeds, boolean):
        self._call_seeds = seeds
        self._call_boolean = boolean


class DummySolver3(BaseSolver):
    def __init__(self):
        self.Left = Component()
        self.Middle = Component()
        self.Right = Component()
    def _get_connected(self, seed, error):
        return (self.Left, self.Middle) if seed == self.Left else (self.Middle, self.Right)


class TestBaseSolver(TestCase):

    def test___init__(self):
        # Test invalid parameters
        with self.assertRaises(TypeError):
            s = BaseSolver(fluid=123, seeds=[])
        with self.assertRaises(TypeError):
            s = BaseSolver(fluid=None, seeds='xzy')
        with self.assertRaises(TypeError):
            s = BaseSolver(fluid=None, seeds=[Component(), 'abc'])
        # Test valid parameters
        f = BaseFluid()
        s = BaseSolver(None, None)
        self.assertEqual(None, s._fluid)
        self.assertEqual([], s._seeds)
        s = BaseSolver(f, [])
        self.assertEqual(f, s._fluid)
        self.assertEqual([], s._seeds)
        c = Component()
        s = BaseSolver(f, c)
        self.assertEqual(f, s._fluid)
        self.assertCountEqual([c], s._seeds)
        c = [Component(), Component(), Component()]
        s = BaseSolver(None, c)
        self.assertEqual(None, s._fluid)
        self.assertCountEqual(c, s._seeds)

    def test_fluid(self):
        # Test setter
        s = BaseSolver(None, None)
        with self.assertRaises(TypeError):
            s.fluid = 123
        f = BaseFluid()
        s.fluid = f
        self.assertEqual(f, s._fluid)
        # Test getter
        s = BaseSolver(None, None)
        self.assertEqual(None, s.fluid)
        s._fluid = 123
        self.assertEqual(123, s.fluid)

    def test_disconnected(self):
        s = DummySolver(True)
        self.assertEqual(False, s.disconnected)
        s = DummySolver(False)
        self.assertEqual(True, s.disconnected)

    def test_seeds(self):
        # Test setter
        s = BaseSolver(None, None)
        with self.assertRaises(TypeError):
            s.seeds = 'abc'
        with self.assertRaises(TypeError):
            s.seeds = [Component(), Component(), 123]
        c = Component()
        s.seeds = c
        self.assertCountEqual([c], s._seeds)
        c = [Component(), Component()]
        s.seeds = c
        self.assertCountEqual(c, s._seeds)
        # Test getter
        s = BaseSolver(None, None)
        self.assertEqual([], s.seeds)
        s._seeds = '567'
        self.assertEqual('567', s.seeds)

    def test__get_connected(self):
        # Test disconnected
        s = BaseSolver()
        c = Component()
        c._add_connector(Connector(c,[]))
        with self.assertRaises(AssertionError):
            s._get_connected(c, True)
        self.assertCountEqual([c], s._get_connected(c, False))
        # Build some system
        c1 = Component()
        c._add_connector(Connector(c1, []))
        c2 = Component()
        c._add_connector(Connector(c2, []))
        c._add_connector(Connector(c2, []))
        c3 = Component()
        con3 = Connector(c3, [])
        c._add_connector(con3)
        c1.connect(c2.connectors[0])
        c3.connect(c2)
        l1 = s._get_connected(c1, True)
        l2 = s._get_connected(c3, True)
        self.assertCountEqual(l1, l2)
        self.assertCountEqual([c1,c2,c3], l1)
        # Test strange error
        con3._parent = 123
        with self.assertRaises(TypeError):
            s._get_connected(c1, True)

    def test__get_connected_list(self):
        s = DummySolver3()
        l = s._get_connected_list([s.Left, s.Right])
        self.assertEqual(3, len(l))
        self.assertCountEqual([s.Left, s.Middle, s.Right], l)

    def test_components(self):
        s = DummySolver2()
        c1 = Component()
        c2 = Component()
        s._seeds = [c1,c2]
        s.components
        self.assertEqual(False, s._call_boolean)
        self.assertCountEqual([c1,c2], s._call_seeds)

    def test_solve(self):
        BaseSolver().solve(0.1, 100.0, 3, 0)
