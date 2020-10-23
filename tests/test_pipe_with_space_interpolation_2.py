from unittest import TestCase
from cavsim.pipes.pipe_with_space_interpolation_2 import Pipe as Pipe
import cavsim
import numpy as np
from numpy import testing as npt


class TestPipe(TestCase):

    def setUp(self) -> None:

        self.pipe = Pipe(diameter=1.0,
                         length=10.0,
                         wall_thickness=0.1,
                         bulk_modulus=2e11,
                         roughness=1e-4,
                         inner_points=3,
                         initial_pressure=1.28e5)

    def tearDown(self) -> None:
        del self.pipe

    def test_length(self):
        self.assertEqual(10.0, self.pipe.length)

    def test_diameter(self):
        self.assertEqual(1.0, self.pipe.diameter)

    def test_wall_thickness(self):
        self.assertEqual(0.1, self.pipe.wall_thickness)

    def test__calculate_space_interpolation(self):
        self.pipe.fields_resize(5)
        self.assertEqual(5, self.pipe.field_wide_slice('pressure', 0)[:].shape[0])
        self.pipe.field_wide_slice('velocity', 1)[:] = np.arange(5)


    def test__calculate_speed_of_sound(self):
        None

    def test__calculate_reynolds(self):
        None

    def test__calculate_darcy_friction_factor(self):
        None

    def test__calculate_friction_steady(self):
        None

    def test__calculate_friction(self):
        None

    def test__calculate_pressure(self):
        None

    def test__calculate_velocity(self):
        None

    def test__calculate_brunone(self):
        None

    def test__calculate_unsteady_friction_a(self):
        None

    def test__calculate_unsteady_friction_b(self):
        None