from unittest import TestCase
from cavsim.base.fluids.base_fluid import BaseFluid


class DummyFluid(BaseFluid):
    def __init__(self, one, two, three=5):
        super(DummyFluid, self).__init__(1, 2, 3, 4)
        self._one = one
        self._two = two
        self._three = three

    def viscosity(self, *args, **kwargs):
        return self._one

    def density(self, *args, **kwargs):
        return self._two

    def bulk_modulus(self, *args, **kwargs):
        return self._three


class TestBaseFluid(TestCase):

    def setUp(self):
        self.fluid = BaseFluid(1, 2, 3, 4)

    def tearDown(self):
        del self.fluid
        self.fluid = None

    def test___init__(self):
        # Invalid parameter tests
        with self.assertRaises(TypeError):
            BaseFluid('abc', 2, 3, 4, 5, 6)
        with self.assertRaises(TypeError):
            BaseFluid(1, 'def', 3, 4, 5, 6)
        with self.assertRaises(TypeError):
            BaseFluid(1, 2, 'ghi', 4, 5, 6)
        with self.assertRaises(TypeError):
            BaseFluid(1, 2, 3, 'jkl', 5, 6)
        with self.assertRaises(TypeError):
            BaseFluid(1, 2, 3, 4, 'mno', 6)
        with self.assertRaises(TypeError):
            BaseFluid(1, 2, 3, 4, 5, 'pqr')
        # Valid parameter tests
        self.assertEqual(101325, self.fluid._norm_pressure)
        self.assertEqual(293.15, self.fluid._norm_temperature)
        f = BaseFluid(9, 8, 7, 6, 5, 4)
        self.assertEqual(9, f._norm_density)
        self.assertEqual(8, f._norm_viscosity)
        self.assertEqual(7, f._norm_bulk_modulus)
        self.assertEqual(6, f._norm_vapor_pressure)
        self.assertEqual(5, f._norm_pressure)
        self.assertEqual(4, f._norm_temperature)

    def test_norm_pressure(self):
        self.fluid._norm_pressure = None
        self.assertEqual(None, self.fluid.norm_pressure)
        self.fluid._norm_pressure = 123.45
        self.assertEqual(123.45, self.fluid.norm_pressure)

    def test_norm_temperature(self):
        self.fluid._norm_temperature = None
        self.assertEqual(None, self.fluid.norm_temperature)
        self.fluid._norm_temperature = 123.45
        self.assertEqual(123.45, self.fluid.norm_temperature)

    def test_norm_density(self):
        self.fluid._norm_density = None
        self.assertEqual(None, self.fluid.norm_density)
        self.fluid._norm_density = 123.45
        self.assertEqual(123.45, self.fluid.norm_density)

    def test_norm_viscosity(self):
        self.fluid._norm_viscosity = None
        self.assertEqual(None, self.fluid.norm_viscosity)
        self.fluid._norm_viscosity = 123.45
        self.assertEqual(123.45, self.fluid.norm_viscosity)

    def test_norm_bulk_modulus(self):
        self.fluid._norm_bulk_modulus = None
        self.assertEqual(None, self.fluid.norm_bulk_modulus)
        self.fluid._norm_bulk_modulus = 123.45
        self.assertEqual(123.45, self.fluid.norm_bulk_modulus)

    def test_norm_compressibility(self):
        self.fluid._norm_bulk_modulus = 123.45
        self.assertAlmostEqual(1.0 / 123.45, self.fluid.norm_compressibility)

    def test_norm_vapor_pressure(self):
        self.fluid._norm_vapor_pressure = None
        self.assertEqual(None, self.fluid.norm_vapor_pressure)
        self.fluid._norm_vapor_pressure = 123.45
        self.assertEqual(123.45, self.fluid.norm_vapor_pressure)

    def test_density(self):
        self.assertEqual(self.fluid.norm_density, self.fluid.density(77, 88))

    def test_viscosity(self):
        self.assertEqual(self.fluid.norm_viscosity, self.fluid.viscosity(99, 111))

    def test_kinematic_viscosity(self):
        f = DummyFluid(1, 2)
        self.assertEqual(0.5, f.kinematic_viscosity(1, 2, 3))
        f = DummyFluid(15, 5)
        self.assertEqual(3, f.kinematic_viscosity(1, 2, 3))

    def test_compressibility(self):
        self.assertEqual(self.fluid.norm_compressibility, self.fluid.compressibility(222))

    def test_vapor_pressure(self):
        self.assertEqual(self.fluid.norm_vapor_pressure, self.fluid.vapor_pressure(333))

    def test_norm_speed_of_sound(self):
        f = BaseFluid(4.0, 0.0, 4.0, 0.0)
        self.assertEqual(1.0, f.norm_speed_of_sound)
        f = BaseFluid(4.0, 0.0, 16.0, 0.0)
        self.assertEqual(2.0, f.norm_speed_of_sound)
        f = BaseFluid(64.0, 0.0, 16.0, 0.0)
        self.assertEqual(0.5, f.norm_speed_of_sound)

    def test_speed_of_sound(self):
        f = DummyFluid(0., 4.0, 4.0)
        self.assertEqual(1.0, f.speed_of_sound(7, 8))
        f = DummyFluid(0., 4.0, 16.0)
        self.assertEqual(2.0, f.speed_of_sound(7, 8))
        f = DummyFluid(0., 64.0, 16.0)
        self.assertEqual(0.5, f.speed_of_sound(7, 8))
