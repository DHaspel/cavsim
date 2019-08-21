from unittest import TestCase
from cavsim.base.fluids.base_fluid import BaseFluid


class TestBaseFluid(TestCase):

    def setUp(self):
        self.fluid = BaseFluid(1, 2, 3, 4)

    def tearDown(self):
        del self.fluid
        self.fluid = None

    def test___init__(self):
        self.assertEqual(101325, self.fluid._norm_pressure)
        self.assertEqual(293.15, self.fluid._norm_temperature)
        f = BaseFluid(9, 8, 7, 6, 5, 4)
        self.assertEqual(9, f._norm_density)
        self.assertEqual(8, f._norm_viscosity)
        self.assertEqual(7, f._norm_compressibility)
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

    def test_norm_compressibility(self):
        self.fluid._norm_compressibility = None
        self.assertEqual(None, self.fluid.norm_compressibility)
        self.fluid._norm_compressibility = 123.45
        self.assertEqual(123.45, self.fluid.norm_compressibility)

    def test_norm_vapor_pressure(self):
        self.fluid._norm_vapor_pressure = None
        self.assertEqual(None, self.fluid.norm_vapor_pressure)
        self.fluid._norm_vapor_pressure = 123.45
        self.assertEqual(123.45, self.fluid.norm_vapor_pressure)

    def test_density(self):
        self.assertEqual(self.fluid.norm_density, self.fluid.density(77, 88))

    def test_viscosity(self):
        self.assertEqual(self.fluid.norm_viscosity, self.fluid.viscosity(99, 111))

    def test_compressibility(self):
        self.assertEqual(self.fluid.norm_compressibility, self.fluid.compressibility(222))

    def test_vapor_pressure(self):
        self.assertEqual(self.fluid.norm_vapor_pressure, self.fluid.vapor_pressure(333))
