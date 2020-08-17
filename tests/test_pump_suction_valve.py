from unittest import TestCase
from cavsim.boundaries.pump_suction_valve import PumpSuctionValve as PumpSuctionValve
import cavsim
import numpy as np
from numpy import testing as npt

class TestPumpSuctionValve(TestCase):

    def setUp(self):
        self.valve = PumpSuctionValve(7950.0,
                                 3.438,
                                 3438.0,
                                 0.022,
                                 0.068,
                                 0.040,
                                 0.03644173,
                                 np.pi/2.0,
                                 7.63e-3,
                                 6.480,
                                 107.00,
                                 74.00,
                                 1.40,
                                 -2.40,
                                 0.50,
                                 0.0,
                                 0.0,
                                 25.0e-3,
                                 20e5,
                                 20e5)

    def tearDown(self) -> None:
        del self.valve

    def test_valve_density(self):
        self.assertEqual(7950, self.valve.valve_density)
        self.assertEqual(7950, self.valve._valve_density)
        self.valve._valve_density = 1000
        self.assertEqual(1000, self.valve.valve_density)

    def test_spring_stiffness(self):
        self.assertEqual(3438.0, self.valve.spring_stiffness)
        self.assertEqual(3438.0, self.valve._spring_stiffness)
        self.valve._spring_stiffness=1000
        self.assertEqual(1000, self.valve._spring_stiffness)

    def test_spring_force0(self):
        self.assertEqual(3.438, self.valve.spring_force0)
        self.assertEqual(3.438, self.valve._spring_force0)
        self.valve._spring_force0 = 1000
        self.assertEqual(1000, self.valve._spring_force0)

    def test_valve_mass(self):
        self.assertEqual(0.068, self.valve.valve_mass)
        self.assertEqual(0.068, self.valve._valve_mass)
        self.valve._valve_mass = 1000
        self.assertEqual(1000, self.valve._valve_mass)

    def test_spring_mass(self):
        self.assertEqual(0.022, self.valve.spring_mass)
        self.assertEqual(0.022, self.valve._spring_mass)
        self.valve._spring_mass = 1000
        self.assertEqual(1000, self.valve._spring_mass)

    def test_outer_diameter(self):
        self.assertEqual(0.04, self.valve.outer_diameter)
        self.assertEqual(0.04, self.valve._outer_diameter)
        self.valve._outer_diameter = 1000
        self.assertEqual(1000, self.valve._outer_diameter)

    def test_inner_diameter(self):
        self.assertEqual(0.03644173, self.valve.inner_diameter)
        self.assertEqual(0.03644173, self.valve._inner_diameter)
        self.valve._inner_diameter = 1000
        self.assertEqual(1000, self.valve._inner_diameter)

    def test_valve_area(self):
        self.assertAlmostEqual(0.00125664, self.valve.valve_area, delta=1e-5)

    def test_radius(self):
        self.assertEqual(0.02, np.max(self.valve.radius))
        self.assertAlmostEqual(0.03644176/2.0, np.min(self.valve.radius), delta=1e-5)

    def test_inner_radius(self):
        self.assertAlmostEqual(0.03644176/2.0, self.valve.inner_radius, delta=1e-5)

    def test_outer_radius(self):
        self.assertEqual(0.02, self.valve.outer_radius)

    def test_contact_area(self):
        self.assertAlmostEqual(0.0002136, self.valve.contact_area, delta=1e-4)

    def test_flow_constant_1(self):
        self.assertEqual(7.63e-3, self.valve.flow_constant_1)
        self.assertEqual(7.63e-3, self.valve._flow_constant_1)
        self.valve._flow_constant_1 = 1000
        self.assertEqual(1000, self.valve._flow_constant_1)

    def test_flow_constant_2(self):
        self.assertEqual(6.480, self.valve.flow_constant_2)
        self.assertEqual(6.480, self.valve._flow_constant_2)
        self.valve._flow_constant_2 = 1000
        self.assertEqual(1000, self.valve._flow_constant_2)

    def test_friction_factor_a(self):
        self.assertEqual(107.0, self.valve.friction_factor_a)
        self.assertEqual(107.0, self.valve._friction_factor_a)
        self.valve._friction_factor_a = 1000
        self.assertEqual(1000, self.valve._friction_factor_a)

    def test_friction_factor_b(self):
        self.assertEqual(74.0, self.valve.friction_factor_b)
        self.assertEqual(74.0, self.valve._friction_factor_b)
        self.valve._friction_factor_b = 1000
        self.assertEqual(1000.0, self.valve._friction_factor_b)

    def test_friction_factor_c(self):
        self.assertEqual(1.40, self.valve.friction_factor_c)
        self.assertEqual(1.40, self.valve._friction_factor_c)
        self.valve._friction_factor_c = 1000
        self.assertEqual(1000, self.valve._friction_factor_c)

    def test_friction_factor_d(self):
        self.assertEqual(-2.40, self.valve.friction_factor_d)
        self.assertEqual(-2.40, self.valve._friction_factor_d)
        self.valve._friction_factor_d = 1000
        self.assertEqual(1000, self.valve._friction_factor_d)

    def test_factor_k0(self):
        self.assertEqual(0.50, self.valve.factor_k0)
        self.assertEqual(0.50, self.valve._factor_k0)
        self.valve._factor_k0 = 1000
        self.assertEqual(1000, self.valve._factor_k0)

    def test_factor_k1(self):
        self.assertEqual(0.0, self.valve.factor_k1)
        self.assertEqual(0.0, self.valve._factor_k1)
        self.valve._factor_k1 = 1000
        self.assertEqual(1000.0, self.valve._factor_k1)

    def test_factor_k2(self):
        self.assertEqual(0.0, self.valve.factor_k2)
        self.assertEqual(0.0, self.valve._factor_k2)
        self.valve._factor_k2 = 1000
        self.assertEqual(1000, self.valve._factor_k2)

    def test_max_displacement(self):
        self.assertEqual(25e-3, self.valve.max_displacement)
        self.assertEqual(25e-3, self.valve._max_displacement)
        self.valve._max_displacement = 1000
        self.assertEqual(1000, self.valve._max_displacement)

    def test_get_max_delta_t(self):
        None

    def test_discretize(self):
        None

    def test_initialize(self):
        # Internal state of component
        npt.assert_equal(0.0, self.valve.field('velocity'))
        npt.assert_equal(0.0, self.valve.field('friction'))
        npt.assert_equal(0.0, self.valve.field('speed_of_sound'))

        # Valve parameters
        npt.assert_equal(0.0, self.valve.field('displacement'))
        npt.assert_equal(0.0, self.valve.field('acceleration'))
        npt.assert_equal(0.0, self.valve.field('valve_velocity'))
        npt.assert_equal(1.0, self.valve.field('valve_zeta'))
        npt.assert_equal(0.0, self.valve.field('spring_force'))
        npt.assert_equal(0.0, self.valve.field('gravity_force'))
        npt.assert_equal(0.0, self.valve.field('acceleration_force'))
        npt.assert_equal(0.0, self.valve.field('damping_force'))
        npt.assert_equal(0.0, self.valve.field('upper_pressure_force'))
        npt.assert_equal(0.0, self.valve.field('lower_pressure_force'))
        npt.assert_equal(0.0, self.valve.field('contact_pressure_force'))
        npt.assert_equal(0.0, self.valve.field('flow_force'))
        npt.assert_equal(0.0, self.valve.field('damping_force'))
        npt.assert_equal(20e5, self.valve.field('upper_pressure'))
        npt.assert_equal(1.28e5, self.valve.field('lower_pressure'))
        npt.assert_equal(0.0, self.valve.field('gap_area'))
        npt.assert_equal(0.0, self.valve.field('delta_p'))
        npt.assert_equal(0.0, self.valve.field('volume_flow'))



    def test_prepare_next_timestep(self):
        self.fail()

    def test_exchange_last_boundaries(self):
        self.fail()

    def test_prepare_next_inner_iteration(self):
        self.fail()

    def test_calculate_gravity_force(self):
        self.assertAlmostEqual(0.7718434, self.valve.calculate_gravity_force(1000), delta=1e-5)

    def test_calculate_spring_force(self):
        self.assertAlmostEqual(37.818, self.valve.calculate_spring_force(0.01), delta=1e-3)

    def test_calculate_lower_pressure_force(self):
        self.fail()

    def test_calculate_upper_pressure_force(self):
        self.fail()

    def test_calculate_contact_pressure(self):
        self.fail()

    def test_calculate_dampening_force(self):
        self.assertAlmostEqual(7.63648, self.valve.calculate_dampening_force(1000, 1, 1e-6), delta=1e-3)
        self.assertAlmostEqual(-7.63648, self.valve.calculate_dampening_force(1000, -1, 1e-6), delta=1e-3)

    def test_calculate_gap_reynolds_number(self):
        self.fail()

    def test_calculate_dimensionless_coefficient_of_force(self):
        self.assertEqual(0.50, self.valve.calculate_dimensionless_coefficient_of_force(1000, 0.01))
        self.valve._factor_k1 = 1.0
        self.valve._outer_diameter = 1.0
        self.assertEqual(0.49, self.valve.calculate_dimensionless_coefficient_of_force(1000, 0.01))

    def test_calculate_flow_force(self):
        self.fail()

    def test_calculate_zeta_value(self):
        self.fail()

    def test_calculate_flow(self):
        self.fail()

    def test_calculate_valve(self):
        self.fail()

    def test_calculate_next_inner_iteration(self):
        self.fail()
