from unittest import TestCase
from cavsim.pipes.pipe_with_space_interpolation_2 import Pipe as Pipe
import cavsim
import numpy as np
from numpy import testing as npt
from cavsim import Measure
from cavsim.connectors import BaseConnector, Connector
from cavsim.channels import ImportChannel, ExportChannel
from cavsim.components import BaseComponent, Component
from cavsim.solvers import BaseSolver, SimpleSolver
from cavsim.fluids import Fluid


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

        fluid = Fluid(1000, 1e-3, 2.08e9, 2.3e3, initial_pressure=1e5)
        solver = SimpleSolver
        solver.fluid = fluid
        self.pipe.fields_resize(5)
        self.assertEqual(5, self.pipe.field_wide_slice('pressure', 0)[:].shape[0])
        self.pipe.field_wide_slice('velocity', 1)[:] = np.ones(5)*0.1+1
        self.pipe.field_wide_slice('pressure', 1)[:] = np.arange(5)*1+1e5
        #npt.assert_equal(self.pipe.field_wide_slice('pressure', 1)[:], 1e5)
        self.pipe._delta_x = 1e-1
        self.pipe._delta_t = 1e-4
        self.pipe._calculate_space_interpolation()
        print('Interpolated delta_x_a\n' + str(self.pipe.field_wide_slice('delta_x_a', 1)[:]))
        print('Interpolated delta_x_b\n' + str(self.pipe.field_wide_slice('delta_x_b', 1)[:]))
        print('Interpolated pressure_a\n' + str(self.pipe.field_wide_slice('pressure_a', 1)[:]))
        print('Interpolated velocity_a\n' + str(self.pipe.field_wide_slice('velocity_a', 1)[:]))
        print('Interpolated pressure_b\n' + str(self.pipe.field_wide_slice('pressure_b', 1)[:]))
        print('Interpolated velocity_b\n' + str(self.pipe.field_wide_slice('velocity_b', 1)[:]))
        print('Original pressure field\n' + str(self.pipe.field_wide_slice('pressure', 1)[:]))

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