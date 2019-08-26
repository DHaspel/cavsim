from unittest import TestCase
import numpy as np
import numpy.testing as npt
from cavsim.base.fluids.fluid import Fluid


class TestFluid(TestCase):

    def setUp(self):
        self.fluid = Fluid(1, 2, 3, 4, 5, 6)

    def tearDown(self):
        del self.fluid
        self.fluid = None

    def test___init__(self):
        f = Fluid(1, 2, 3, 4, 5, 6)
        self.assertEqual(None, f._density_cb)
        self.assertEqual(None, f._viscosity_cb)
        self.assertEqual(None, f._bulk_modulus_cb)
        self.assertEqual(None, f._vapor_pressure_cb)

    def test__density(self):
        self.assertEqual(self.fluid.norm_density, self.fluid._density(1, 2))
        answer = np.asarray([self.fluid.norm_density, self.fluid.norm_density])
        result = self.fluid._density([1, 1], [2, 2])
        self.assertEqual(answer.shape, result.shape)
        npt.assert_almost_equal(result, answer)

    def test__viscosity(self):
        self.assertEqual(self.fluid.norm_viscosity, self.fluid._viscosity(3, 4))
        answer = np.asarray([self.fluid.norm_viscosity, self.fluid.norm_viscosity])
        result = self.fluid._viscosity([3, 3], [4, 4])
        self.assertEqual(answer.shape, result.shape)
        npt.assert_allclose(result, answer)

    def test__bulk_modulus(self):
        self.assertEqual(self.fluid.norm_bulk_modulus, self.fluid._bulk_modulus(5))
        answer = np.asarray([self.fluid.norm_bulk_modulus, self.fluid.norm_bulk_modulus])
        result = self.fluid._bulk_modulus([5, 5])
        self.assertEqual(answer.shape, result.shape)
        npt.assert_allclose(result, answer)

    def test__vapor_pressure(self):
        self.assertEqual(self.fluid.norm_vapor_pressure, self.fluid._vapor_pressure(6))
        answer = np.asarray([self.fluid.norm_vapor_pressure, self.fluid.norm_vapor_pressure])
        result = self.fluid._vapor_pressure([6, 6])
        self.assertEqual(answer.shape, result.shape)
        npt.assert_allclose(result, answer)

    def test_density(self):
        self.assertEqual(self.fluid.norm_density, self.fluid.density(1, 2))
        self.fluid._density_cb = lambda x, y, z: 99
        self.assertEqual(99, self.fluid.density(1, 2))

    def test_viscosity(self):
        self.assertEqual(self.fluid.norm_viscosity, self.fluid.viscosity(3, 4))
        self.fluid._viscosity_cb = lambda x, y, z: 88
        self.assertEqual(88, self.fluid.viscosity(3, 4))

    def test_bulk_modulus(self):
        self.assertEqual(self.fluid.norm_bulk_modulus, self.fluid.bulk_modulus(5))
        self.fluid._bulk_modulus_cb = lambda x, y: 77
        self.assertEqual(77, self.fluid.bulk_modulus(5))

    def test_vapor_pressure(self):
        self.assertEqual(self.fluid.norm_vapor_pressure, self.fluid.vapor_pressure(6))
        self.fluid._vapor_pressure_cb = lambda x, y: 66
        self.assertEqual(66, self.fluid.vapor_pressure(6))

    def test_density_cb(self):
        # Test getter
        cb = lambda x, y, z: 99
        self.assertEqual(None, self.fluid.density_cb)
        self.fluid._density_cb = cb
        self.assertEqual(cb, self.fluid.density_cb)
        # Test setter
        self.fluid.density_cb = None
        self.assertEqual(None, self.fluid._density_cb)
        self.fluid.density_cb = cb
        self.assertEqual(cb, self.fluid._density_cb)
        with self.assertRaises(TypeError):
            self.fluid.density_cb = 123

    def test_viscosity_cb(self):
        # Test getter
        cb = lambda x, y, z: 99
        self.assertEqual(None, self.fluid.viscosity_cb)
        self.fluid._viscosity_cb = cb
        self.assertEqual(cb, self.fluid.viscosity_cb)
        # Test setter
        self.fluid.viscosity_cb = None
        self.assertEqual(None, self.fluid._viscosity_cb)
        self.fluid.viscosity_cb = cb
        self.assertEqual(cb, self.fluid._viscosity_cb)
        with self.assertRaises(TypeError):
            self.fluid.viscosity_cb = 123

    def test_bulk_modulus_cb(self):
        # Test getter
        cb = lambda x, y: 99
        self.assertEqual(None, self.fluid.bulk_modulus_cb)
        self.fluid._bulk_modulus_cb = cb
        self.assertEqual(cb, self.fluid.bulk_modulus_cb)
        # Test setter
        self.fluid.bulk_modulus_cb = None
        self.assertEqual(None, self.fluid._bulk_modulus_cb)
        self.fluid.bulk_modulus_cb = cb
        self.assertEqual(cb, self.fluid._bulk_modulus_cb)
        with self.assertRaises(TypeError):
            self.fluid.bulk_modulus_cb = 123

    def test_vapor_pressure_cb(self):
        # Test getter
        cb = lambda x, y: 99
        self.assertEqual(None, self.fluid.vapor_pressure_cb)
        self.fluid._vapor_pressure_cb = cb
        self.assertEqual(cb, self.fluid.vapor_pressure_cb)
        # Test setter
        self.fluid.vapor_pressure_cb = None
        self.assertEqual(None, self.fluid._vapor_pressure_cb)
        self.fluid.vapor_pressure_cb = cb
        self.assertEqual(cb, self.fluid._vapor_pressure_cb)
        with self.assertRaises(TypeError):
            self.fluid.vapor_pressure_cb = 123
