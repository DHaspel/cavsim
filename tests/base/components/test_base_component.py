from unittest import TestCase
from cavsim.base.components.base_component import BaseComponent
from cavsim.base.fluids.base_fluid import BaseFluid


class TestBaseComponent(TestCase):

    def test___init__(self):
        c = BaseComponent()
        self.assertEqual(None, c._fluid)
        self.assertEqual(None, c._global_fluid)

    def test_fluid(self):
        # Test getter
        c = BaseComponent()
        with self.assertRaises(AssertionError):
            c.fluid
        c._fluid = 456
        self.assertEqual(456, c.fluid)
        c._fluid = None
        c._global_fluid = 'abc'
        self.assertEqual('abc', c.fluid)
        # Test setter
        c = BaseComponent()
        f = BaseFluid(1, 2, 3, 4)
        self.assertEqual(None, c._fluid)
        with self.assertRaises(TypeError):
            c.fluid = 123
        c.fluid = f
        self.assertEqual(f, c._fluid)

    def test_get_max_delta_t(self):
        self.assertEqual(None, BaseComponent().get_max_delta_t())

    def test_check_fluid(self):
        c = BaseComponent()
        f = BaseFluid(1, 2, 3, 4)
        self.assertEqual(None, c._global_fluid)
        c.check_fluid(123)
        self.assertEqual(123, c._global_fluid)
        c.check_fluid(f)
        self.assertEqual(f, c._global_fluid)

    def test_discretize(self):
        BaseComponent().discretize(0.1)

    def test_initialize(self):
        BaseComponent().initialize()

    def test_prepare_next_timestep(self):
        BaseComponent().prepare_next_timestep(0.1, 100.0)

    def test_prepare_next_inner_iteration(self):
        BaseComponent().prepare_next_inner_iteration(1)

    def test_calculate_next_inner_iteration(self):
        self.assertEqual(False, BaseComponent().calculate_next_inner_iteration(1))

