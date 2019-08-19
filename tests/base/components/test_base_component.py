from unittest import TestCase
from cavsim.base.components.base_component import BaseComponent
from cavsim.measure import Measure
from cavsim.base.fluids.base_fluid import BaseFluid


class TestBaseComponent(TestCase):

    def test___init__(self):
        c = BaseComponent()
        self.assertEqual(c._fluid, None)
        self.assertEqual(c._global_fluid, None)

    def test_fluid(self):
        # Test getter
        c = BaseComponent()
        self.assertEqual(c.fluid, None)
        c._fluid = 456
        self.assertEqual(c.fluid, 456)
        c._fluid = None
        c._global_fluid = 'abc'
        self.assertEqual(c.fluid, 'abc')
        # Test setter
        c = BaseComponent()
        f = BaseFluid()
        self.assertEqual(c._fluid, None)
        with self.assertRaises(TypeError):
            c.fluid = 123
        c.fluid = f
        self.assertEqual(c._fluid, f)

    def test_get_max_delta_t(self):
        self.assertEqual(BaseComponent().get_max_delta_t(), None)

    def test_check_fluid(self):
        c = BaseComponent()
        f = BaseFluid()
        self.assertEqual(c._global_fluid, None)
        c.check_fluid(123)
        self.assertEqual(c._global_fluid, 123)
        c.check_fluid(f)
        self.assertEqual(c._global_fluid, f)

    def test_discretize(self):
        BaseComponent().discretize(0.1)

    def test_initialize(self):
        BaseComponent().initialize()

    def test_prepare_next_timestep(self):
        BaseComponent().prepare_next_timestep(0.1, 100.0)

    def test_prepare_next_inner_iteration(self):
        BaseComponent().prepare_next_inner_iteration(1)

    def test_calculate_next_inner_iteration(self):
        self.assertEqual(BaseComponent().calculate_next_inner_iteration(1), False)

