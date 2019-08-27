from unittest import TestCase
import numpy as np
import numpy.testing as npt
from cavsim.base.fluids.base_fluid import BaseFluid


class DummyFluid(BaseFluid):
    def __init__(self, one, two, three=5):
        super(DummyFluid, self).__init__(1, 2, 3, 4)
        self._one = one
        self._two = two
        self._three = three

    def viscosity(self, *args, **kwargs):
        return self._one * self._ones(*args)

    def density(self, *args, **kwargs):
        return self._two * self._ones(*args)

    def bulk_modulus(self, *args, **kwargs):
        return self._three * self._ones(*args)


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
        p = self.fluid.norm_pressure
        t = self.fluid.norm_temperature
        self.assertEqual(self.fluid.norm_density, self.fluid.density(p, t))
        answer = np.asarray([self.fluid.norm_density, self.fluid.norm_density])
        result = self.fluid.density([p, p], [t, t])
        self.assertEqual(answer.shape, result.shape)
        npt.assert_almost_equal(result, answer)
        self.assertEqual(self.fluid.norm_density, self.fluid.density())
        f = BaseFluid(2.0, 0, 3.0, 0)
        p = f.norm_pressure + 5.0
        self.assertAlmostEqual(10.5889801009, f.density(p))

    def test_viscosity(self):
        self.assertEqual(self.fluid.norm_viscosity, self.fluid.viscosity(99, 111))
        answer = np.asarray([self.fluid.norm_viscosity, self.fluid.norm_viscosity])
        result = self.fluid.viscosity([99, 99], [111, 111])
        self.assertEqual(answer.shape, result.shape)
        npt.assert_allclose(result, answer)

    def test_kinematic_viscosity(self):
        f = DummyFluid(1, 2)
        self.assertEqual(0.5, f.kinematic_viscosity(1, 2, 3))
        f = DummyFluid(15, 5)
        self.assertEqual(3, f.kinematic_viscosity(1, 2, 3))
        answer = np.asarray([3, 3])
        result = f.kinematic_viscosity([1, 1], [2, 2], [3, 3])
        self.assertEqual(answer.shape, result.shape)
        npt.assert_allclose(result, answer)

    def test_compressibility(self):
        self.assertEqual(self.fluid.norm_compressibility, self.fluid.compressibility(222))
        answer = np.asarray([self.fluid.norm_compressibility, self.fluid.norm_compressibility])
        result = self.fluid.compressibility([222, 222])
        self.assertEqual(answer.shape, result.shape)
        npt.assert_allclose(result, answer)

    def test_bulk_modulus(self):
        self.assertEqual(self.fluid.norm_bulk_modulus, self.fluid.bulk_modulus(222))
        answer = np.asarray([self.fluid.norm_bulk_modulus, self.fluid.norm_bulk_modulus])
        result = self.fluid.bulk_modulus([222, 222])
        self.assertEqual(answer.shape, result.shape)
        npt.assert_allclose(result, answer)

    def test_vapor_pressure(self):
        self.assertEqual(self.fluid.norm_vapor_pressure, self.fluid.vapor_pressure(333))
        answer = np.asarray([self.fluid.norm_vapor_pressure, self.fluid.norm_vapor_pressure])
        result = self.fluid.vapor_pressure([333, 333])
        self.assertEqual(answer.shape, result.shape)
        npt.assert_allclose(result, answer)

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
        # todo: better dummy for numpy replacements
        #answer = np.asarray([0.5, 0.5, 0.5])
        #result = f.speed_of_sound([7, 7, 7], [8, 8, 8])
        #self.assertEqual(answer.shape, result.shape)
        #npt.assert_allclose(result, answer)

    def test__ones(self):
        f = BaseFluid(4.0, 0.0, 4.0, 0.0)
        self.assertEqual(1.0, f._ones())
        self.assertEqual(1.0, f._ones(4))
        self.assertEqual(1.0, f._ones(param2=4))
        self.assertEqual(1.0, f._ones(7, 8))
        npt.assert_almost_equal(np.asarray([1, 1, 1]), f._ones(np.asarray([1, 2, 3])))
        npt.assert_almost_equal(np.asarray([1, 1]), f._ones(np.asarray([1, 2]), np.asarray([3, 4])))
        with self.assertRaises(IndexError):
            f._ones(1, np.asarray([1, 2]))
        with self.assertRaises(IndexError):
            f._ones(np.asarray([1, 2]), np.asarray([3, 4, 5, 6]))
