from unittest import TestCase
import numpy as np
from cavsim.pipes.base_pipe import BasePipe
from cavsim.base.fluids.fluid import Fluid


class DummyFluid(Fluid):
    def _bulk_modulus(self, t):
        return 3.0 * self.norm_bulk_modulus

    def _density(self, p, t):
        return 2.0 * self.norm_density



class TestBasePipe(TestCase):

    def setUp(self):
        self.pipe = BasePipe(1.0, 2.0, 3.0, 4.0, 5.0)

    def tearDown(self):
        del self.pipe
        self.pipe = None

    def test___init__(self):
        # Invalid parameter tests
        with self.assertRaises(TypeError):
            BasePipe('abc', 2.0, 3.0, 4.0, 5.0)
        with self.assertRaises(TypeError):
            BasePipe(1.0, 'def', 3.0, 4.0, 5.0)
        with self.assertRaises(TypeError):
            BasePipe(1.0, 2.0, 'ghi', 4.0, 5.0)
        with self.assertRaises(TypeError):
            BasePipe(1.0, 2.0, 3.0, 'ijk', 5.0)
        with self.assertRaises(TypeError):
            BasePipe(1.0, 2.0, 3.0, 4.0, 'lmn')
        # Valid parameter tests
        p = BasePipe(1.0, 2.0, 3.0, 4.0, 5.0)
        self.assertEqual(1.0, p._diameter)
        self.assertEqual(2.0, p._length)
        self.assertEqual(3.0, p._wall_thickness)
        self.assertEqual(4.0, p._bulk_modulus)
        self.assertEqual(5.0, p._roughness)

    def test_diameter(self):
        # Test setter
        with self.assertRaises(TypeError):
            self.pipe.diameter = 'abc'
        with self.assertRaises(TypeError):
            self.pipe.diameter = [7.6]
        self.pipe.diameter = 99.0
        self.assertEqual(99.0, self.pipe._diameter)
        # Test getter
        self.pipe._diameter = None
        self.assertEqual(None, self.pipe.diameter)
        self.pipe._diameter = 77.0
        self.assertEqual(77.0, self.pipe.diameter)

    def test_length(self):
        # Test setter
        with self.assertRaises(TypeError):
            self.pipe.length = 'abc'
        with self.assertRaises(TypeError):
            self.pipe.length = [7.6]
        self.pipe.length = 99.0
        self.assertEqual(99.0, self.pipe._length)
        # Test getter
        self.pipe._length = None
        self.assertEqual(None, self.pipe.length)
        self.pipe._length = 77.0
        self.assertEqual(77.0, self.pipe.length)

    def test_area(self):
        self.pipe.diameter = 2.0
        self.assertAlmostEqual(3.1415926535, self.pipe.area)
        self.pipe.diameter = 4.0
        self.assertAlmostEqual(12.566370614, self.pipe.area)

    def test_volume(self):
        self.pipe.diameter = 2.0
        self.pipe.length = 1.0
        self.assertAlmostEqual(3.1415926535, self.pipe.volume)
        self.pipe.diameter = 4.0
        self.pipe.length = 5.0
        self.assertAlmostEqual(62.8318530717, self.pipe.volume)

    def test_wall_thickness(self):
        # Test setter
        with self.assertRaises(TypeError):
            self.pipe.wall_thickness = 'abc'
        with self.assertRaises(TypeError):
            self.pipe.wall_thickness = [7.6]
        self.pipe.wall_thickness = 99.0
        self.assertEqual(99.0, self.pipe._wall_thickness)
        # Test getter
        self.pipe._wall_thickness = None
        self.assertEqual(None, self.pipe.wall_thickness)
        self.pipe._wall_thickness = 77.0
        self.assertEqual(77.0, self.pipe.wall_thickness)

    def test_bulk_modulus(self):
        # Test setter
        with self.assertRaises(TypeError):
            self.pipe.bulk_modulus = 'abc'
        with self.assertRaises(TypeError):
            self.pipe.bulk_modulus = [7.6]
        self.pipe.bulk_modulus = 99.0
        self.assertEqual(99.0, self.pipe._bulk_modulus)
        # Test getter
        self.pipe._bulk_modulus = None
        self.assertEqual(None, self.pipe.bulk_modulus)
        self.pipe._bulk_modulus = 77.0
        self.assertEqual(77.0, self.pipe.bulk_modulus)

    def test_roughness(self):
        # Test setter
        with self.assertRaises(TypeError):
            self.pipe.roughness = 'abc'
        with self.assertRaises(TypeError):
            self.pipe.roughness = [7.6]
        self.pipe.roughness = 99.0
        self.assertEqual(99.0, self.pipe._roughness)
        # Test getter
        self.pipe._roughness = None
        self.assertEqual(None, self.pipe.roughness)
        self.pipe._roughness = 77.0
        self.assertEqual(77.0, self.pipe.roughness)

    def test_norm_speed_of_sound(self):
        p = BasePipe(1.0, 2.0, 3.0, 4.0, 5.0)
        p.fluid = DummyFluid(6.0, 7.0, 8.0, 9.0, 10.0)
        self.assertAlmostEqual(np.sqrt(0.8), p.norm_speed_of_sound)

    def test_speed_of_sound(self):
        p = BasePipe(1.0, 2.0, 3.0, 4.0, 5.0)
        p.fluid = DummyFluid(6.0, 7.0, 8.0, 9.0, 10.0)
        self.assertAlmostEqual(np.sqrt(2.0 / 3.0), p.speed_of_sound())
