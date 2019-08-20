import warnings
from contextlib import redirect_stdout
from io import StringIO
from unittest import TestCase
from cavsim.base.solvers.simple_solver import SimpleSolver
from cavsim.base.components.component import Component
from cavsim.base.fluids.base_fluid import BaseFluid
from cavsim.base.connectors.connector import Connector


class DummyComponent(Component):
    def __init__(self, delta_t = None):
        super(DummyComponent, self).__init__()
        self._call_delta_t = delta_t
        self._add_connector(Connector(self,[]))
        self._call_discretize = None
        self._call_initialize = None
        self._call_timestep = 0
        self._call_inner = 0
        self._call_calc = 0
    def check_fluid(self, fluid):
        self._call_fluid = fluid
    def get_max_delta_t(self):
        return self._call_delta_t
    def discretize(self, delta_t):
        self._call_discretize = delta_t
    def initialize(self):
        self._call_initialize = True
    def prepare_next_timestep(self, delta_t, total_time):
        self._call_timestep += 1
    def prepare_next_inner_iteration(self, iter):
        self._call_inner += 1
    def calculate_next_inner_iteration(self, iter):
        self._call_calc += 1
        return True if iter < 4 else False



class WrapperSolver(SimpleSolver):
    def __init__(self):
        super(WrapperSolver, self).__init__()
        self._call_delta_t = None
        self._call_discretize = None
        self._call_inner_loop = None
    def _get_delta_t(self, delta_t):
        self._call_delta_t = True
        return 0.1
    def _discretize(self, delta_t):
        self._call_discretize = delta_t
    def _solve_inner_loop(self, max_iter):
        self._call_inner_loop = True


class TestSimpleSolver(TestCase):

    def test__get_delta_t(self):
        s = SimpleSolver()
        s._fluid = 123
        c1 = DummyComponent(0.3)
        c2 = DummyComponent(7.5)
        c1.connect(c2)
        c3 = DummyComponent(0.01)
        c4 = DummyComponent()
        c3.connect(c4)
        s.seeds = [c1,c4]
        with warnings.catch_warnings(record=True) as w:
            delta_t = s._get_delta_t(0.2)
        self.assertEqual(delta_t, 0.01)
        self.assertEqual(c1._call_fluid, s._fluid)
        self.assertEqual(c2._call_fluid, s._fluid)
        self.assertEqual(c3._call_fluid, s._fluid)
        self.assertEqual(c4._call_fluid, s._fluid)

    def test__discretize(self):
        s = SimpleSolver()
        s._fluid = 123
        c1 = DummyComponent(0.3)
        c2 = DummyComponent(7.5)
        c1.connect(c2)
        c3 = DummyComponent(0.01)
        c4 = DummyComponent()
        c3.connect(c4)
        s.seeds = [c1,c4]
        s._discretize(0.2)
        self.assertEqual(c1._call_discretize, 0.2)
        self.assertEqual(c2._call_discretize, 0.2)
        self.assertEqual(c3._call_discretize, 0.2)
        self.assertEqual(c4._call_discretize, 0.2)
        self.assertEqual(c1._call_initialize, True)
        self.assertEqual(c2._call_initialize, True)
        self.assertEqual(c3._call_initialize, True)
        self.assertEqual(c4._call_initialize, True)

    def test__solve_inner_loop(self):
        s = SimpleSolver()
        c1 = DummyComponent(0.3)
        c2 = DummyComponent(7.5)
        c1.connect(c2)
        s.seeds = c1
        s._solve_inner_loop(10)
        self.assertEqual(c1._call_inner, 5)
        self.assertEqual(c2._call_inner, 5)
        self.assertEqual(c1._call_calc, 5)
        self.assertEqual(c1._call_calc, 5)
        s = SimpleSolver()
        c1 = DummyComponent(0.3)
        c2 = DummyComponent(7.5)
        c1.connect(c2)
        s.seeds = c1
        with warnings.catch_warnings(record=True) as w:
            s._solve_inner_loop(3)
        self.assertEqual(c1._call_inner, 3)
        self.assertEqual(c2._call_inner, 3)
        self.assertEqual(c1._call_calc, 3)
        self.assertEqual(c1._call_calc, 3)

    def test_solve(self):
        # Test invalid parameters
        s = SimpleSolver()
        with self.assertRaises(TypeError):
            s.solve('abc', 1.0, 5, 1)
        with self.assertRaises(TypeError):
            s.solve(-0.25, 1.0, 5, 1)
        with self.assertRaises(TypeError):
            s.solve(0.25, 'xyz', 5, 1)
        with self.assertRaises(TypeError):
            s.solve(0.25, -1.0, 5, 1)
        with self.assertRaises(TypeError):
            s.solve(0.25, 1.0, 'def', 1)
        with self.assertRaises(TypeError):
            s.solve(0.25, 1.0, 5, ['ijk'])
        with self.assertRaises(TypeError):
            s.solve(0.25, 1.0, 5, -1)
        with self.assertRaises(TypeError):
            s.solve(0.25, 1.0, 5, 7)
        # Test valid parameters
        s = WrapperSolver()
        c1 = DummyComponent(0.3)
        c2 = DummyComponent(7.5)
        c1.connect(c2)
        s.seeds = c1
        with redirect_stdout(StringIO()) as stdout:
            s.solve(0.25, 1.0, 5, 1)
        self.assertEqual(s._call_delta_t, True)
        self.assertEqual(s._call_discretize, 0.1)
        self.assertEqual(c1._call_timestep, 11)
        self.assertEqual(c2._call_timestep, 11)
