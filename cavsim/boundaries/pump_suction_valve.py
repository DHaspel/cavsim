#! /opt/conda/bin/python3
""" Pipe class implementing a right boundary with given velocity """

# Copyright 2019 FAU-iPAT (http://ipat.uni-erlangen.de/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Optional, Union
import numpy as np
from .base_boundary import BaseBoundary, BoundaryFunction
from ..base.connectors.connector import Connector
from ..measure import Measure
from ..base.channels.import_channel import ImportChannel
from ..base.channels.export_channel import ExportChannel
from scipy import integrate


class PumpSuctionValve(BaseBoundary):
    """
    Pipe class implementing the pipe simulation calculations
    """

    def __init__(
            self,                    # -
            valve_density: float,           # kg/m³
            spring_force0: float,           # N
            spring_stiffness: float,        # N/m
            spring_mass: float,             # kg
            valve_mass: float,              # kg
            outer_diameter: float,          # m
            inner_diameter: float,          # m
            seat_tilt: float,               # °
            flow_constant_1: float,         # -
            flow_constant_2: float,         # -
            friction_factor_a: float,       # -
            friction_factor_b: float,       # -
            friction_factor_c: float,       # -
            friction_factor_d: float,       # -
            factor_k0: float,               # -
            factor_k1: float,               # -
            factor_k2: float,               # -
            max_displacement,               # m

    ) -> None:

        super(PumpSuctionValve, self).__init__()
        # Register values

        self._valve_density = valve_density
        self._spring_force0 = spring_force0
        self._spring_stiffness = spring_stiffness
        self._outer_diameter = outer_diameter
        self._inner_diameter = inner_diameter
        self._seat_tilt = seat_tilt
        self._valve_mass = valve_mass
        self._spring_mass = spring_mass
        self._flow_constant_1 = flow_constant_1
        self._flow_constant_2 = flow_constant_2
        self._friction_factor_a = friction_factor_a
        self._friction_factor_b = friction_factor_b
        self._friction_factor_c = friction_factor_c
        self._friction_factor_d = friction_factor_d
        self._g = 9.81
        self._factor_k0 = factor_k0
        self._factor_k1 = factor_k1
        self._factor_k2 = factor_k2
        self._max_displacement = max_displacement
        self._cases = []
        self.flow_counter = False
        self.epsilon = []
        self.epsilon.append(0.0)


        # Register internal fields
        self._pressure: np.ndarray = self.field_create('pressure', 3)
        self._upper_pressure: np.ndarray = self.field_create('upper_pressure', 3)
        self._lower_pressure: np.ndarray = self.field_create('lower_pressure', 3)
        self._velocity: np.ndarray = self.field_create('velocity', 3)
        self._volume_flow: np.ndarray = self.field_create('volume_flow', 3)
        self._sos: np.ndarray = self.field_create('speed_of_sound', 3)
        self._friction = self.field_create('friction', 3)
        self._valve_displacement = self.field_create('displacement', 4)
        self._valve_velocity = self.field_create('valve_velocity', 4)
        self._valve_acceleration = self.field_create('acceleration', 4)
        self._spring_force: np.ndarray = self.field_create('spring_force', 3)
        self._gravity_force: np.ndarray = self.field_create('gravity_force', 3)
        self._acceleration_force: np.ndarray = self.field_create('acceleration_force', 3)
        self._damping_force: np.ndarray = self.field_create('damping_force', 3)
        self._upper_pressure_force: np.ndarray = self.field_create('upper_pressure_force', 3)
        self._lower_pressure_force: np.ndarray = self.field_create('lower_pressure_force', 3)
        self._contact_pressure_force: np.ndarray = self.field_create('contact_pressure_force', 3)
        self._flow_force: np.ndarray = self.field_create('flow_force', 3)
        self._valve_zeta = self.field_create('valve_zeta', 3)
        self._gap_area = self.field_create('gap_area', 3)
        self._area = np.empty(2)
        self._delta_p = self.field_create('delta_p', 3)

        # Create the left connector
        self._left: Connector = Connector(self, [
            ImportChannel(Measure.deltaX, False),
            ExportChannel(Measure.boundaryPoint, lambda: True),
            ImportChannel(Measure.pressureLast, False),
            ImportChannel(Measure.pressureLast2, False),
            ExportChannel(Measure.pressureCurrent, lambda: self._pressure[0, 1]),
            ExportChannel(Measure.pressureLast, lambda: self._pressure[1, 1]),
            ImportChannel(Measure.velocityPlusLast, False),
            ExportChannel(Measure.velocityMinusCurrent, lambda: -(self._volume_flow[0, 1]) / (self._area[0])),
            ExportChannel(Measure.velocityMinusLast, lambda: -(self._volume_flow[1, 1]) / (self._area[0])),
            ImportChannel(Measure.frictionLast, False),
            ImportChannel(Measure.BPspeedOfSoundLast, False),
            ImportChannel(Measure.area)
        ])
        # Create the right connector
        self._right: Connector = Connector(self, [
            ImportChannel(Measure.deltaX, False),
            ExportChannel(Measure.boundaryPoint, lambda: True),
            ImportChannel(Measure.pressureLast, False),
            ImportChannel(Measure.pressureLast2, False),
            ExportChannel(Measure.pressureCurrent, lambda: self._pressure[0, -2]),
            ExportChannel(Measure.pressureLast, lambda: self._pressure[1, -2]),
            ImportChannel(Measure.velocityMinusLast, False),
            ExportChannel(Measure.velocityPlusCurrent, lambda: (self._volume_flow[0, 1]) / (self._area[1])),
            ExportChannel(Measure.velocityPlusLast, lambda: (self._volume_flow[1, 1]) / (self._area[1])),
            ImportChannel(Measure.frictionLast, False),
            ImportChannel(Measure.BPspeedOfSoundLast, False),
            ImportChannel(Measure.area)
        ])

    @property
    def left(self) -> Connector:
        """
        Left connector property

        :return: Left sided connector of the pipe
        """
        return self._left

    @property
    def right(self) -> Connector:
        """
        Right connector property

        :return: Right sided connector of the pipe
        """
        return self._right

    @property
    def valve_density(self):

        return self._valve_density

    @property
    def spring_stiffness(self):

        return self._spring_stiffness

    @property
    def spring_force0(self):

        return self._spring_force0

    @property
    def valve_mass(self):

        return self._valve_mass

    @property
    def spring_mass(self):

        return self._spring_mass

    @property
    def outer_diameter(self):

        return self._outer_diameter

    @property
    def inner_diameter(self):

        return self._inner_diameter

    @property
    def valve_area(self):

        return np.pi * np.power(self.outer_diameter, 2) / 4

    @property
    def radius(self):

        return np.linspace(self._inner_radius, self._outer_radius, 10000)

    @property
    def _inner_radius(self):

        return self._inner_diameter / 2.0

    @property
    def _outer_radius(self):

        return self._outer_diameter / 2.0

    @property
    def contact_area(self):

        return np.pi * ((self._outer_diameter / 2)**2 - (self._inner_diameter / 2)**2)

    @property
    def flow_constant_1(self):

        return self._flow_constant_1

    @property
    def flow_constant_2(self):

        return self._flow_constant_2

    @property
    def friction_factor_a(self):

        return self._friction_factor_a

    @property
    def friction_factor_b(self):

        return self._friction_factor_b

    @property
    def friction_factor_c(self):

        return self._friction_factor_c

    @property
    def friction_factor_d(self):

        return self._friction_factor_d

    @property
    def factor_k0(self):

        return self._factor_k0

    @property
    def factor_k1(self):

        return self._factor_k1

    @property
    def factor_k2(self):

        return self._factor_k2

    @property
    def max_displacement(self):

        return self._max_displacement

    def get_max_delta_t(self) -> Optional[float]:
        """
        Method to return the maximum allowed timestep width for this component

        :return: Maximum allowed timestep width or None if any is suitable
        """

    def discretize(self, delta_t: float) -> None:
        """
        Method handling the discretization of the component (for a given timestep width)

        :param delta_t: Timestep width to discretize for
        :raises ValueError: Timestep too large to fit at least 3 inner points
        """
        self._delta_t = delta_t
        self.fields_resize(4)

    def initialize(self) -> None:
        """
        Initialize the internal state of the component (after discretization was called)
        """
        self.field('velocity')[:, :] = np.zeros(self.field('velocity').shape)[:, :]
        self.field('pressure')[:, :] = self.fluid.initial_pressure * np.ones(self.field('pressure').shape)[:, :]
        self.field('friction')[:, :] = np.zeros(self.field('friction').shape)[:, :]
        self.field('speed_of_sound')[:, :] = np.zeros(self.field('friction').shape)[:, :]

        # Initialize Valve parameters

        self.field('displacement')[:, :] = np.zeros(self.field('displacement').shape)[:, :]
        self.field('acceleration')[:, :] = np.zeros(self.field('acceleration').shape)[:, :]
        self.field('valve_velocity')[:, :] = np.zeros(self.field('valve_velocity').shape)[:, :]
        self.field('valve_zeta')[:, :] = np.ones(self.field('valve_zeta').shape)[:, :]
        self.field('spring_force')[:, :] = np.zeros(self.field('spring_force').shape)[:, :] * self._spring_force0
        self.field('gravity_force')[:, :] = np.zeros(self.field('gravity_force').shape)[:, :]
        self.field('acceleration_force')[:, :] = np.zeros(self.field('acceleration_force').shape)[:, :]
        self.field('damping_force')[:, :] = np.zeros(self.field('damping_force').shape)[:, :]
        self.field('upper_pressure_force')[:, :] = np.zeros(self.field('upper_pressure_force').shape)[:, :]
        self.field('lower_pressure_force')[:, :] = np.zeros(self.field('lower_pressure_force').shape)[:, :]
        self.field('contact_pressure_force')[:, :] = np.zeros(self.field('contact_pressure_force').shape)[:, :]
        self.field('flow_force')[:, :] = np.zeros(self.field('flow_force').shape)[:, :]
        self.field('damping_force')[:, :] = np.zeros(self.field('damping_force').shape)[:, :]
        self.field('upper_pressure')[:, :] = self.fluid.initial_pressure * np.ones(self.field('upper_pressure').shape)[:, :]
        self.field('lower_pressure')[:, :] = self.fluid.initial_pressure * np.ones(self.field('lower_pressure').shape)[:, :]
        self.field('gap_area')[:, :] = np.ones(self.field('gap_area').shape)[:, :]*1e-7
        self.field('delta_p')[:, :] = np.zeros(self.field('delta_p').shape)[:, :]

    def prepare_next_timestep(self, delta_t: float, next_total_time: float) -> None:
        """
        Prepare the internal state for the next timestep to be calculated

        :param delta_t: Timestep width for the next timestep
        :param next_total_time: Total simulation time at the end of the next timestep
        """
        # Shift all internal fields
        self.fields_move()
        # Set fixed pressure value

    def exchange_last_boundaries(self) -> None:
        """
        Exchange boundary values from previous time steps
        """
        # Exchange previous values with the left boundary
        self._pressure[1, -1] = self.right.value(Measure.pressureLast)
        self._velocity[1, -1] = -self.right.value(Measure.velocityMinusLast)
        self._friction[1, -1] = self.right.value(Measure.frictionLast)
        self._sos[1, -1] = self.right.value(Measure.BPspeedOfSoundLast)
        self._area[1] = self.right.value(Measure.area)
        self._pressure[2, -1] = self.right.value(Measure.pressureLast2)

        # Exchange previous values with the right boundary
        self._pressure[1, 0] = self.left.value(Measure.pressureLast)
        self._velocity[1, 0] = self.left.value(Measure.velocityPlusLast)
        self._friction[1, 0] = self.left.value(Measure.frictionLast)
        self._sos[1, 0] = self.left.value(Measure.BPspeedOfSoundLast)
        self._area[0] = self.left.value(Measure.area)
        self._pressure[2, 0] = self.left.value(Measure.pressureLast2)

    def prepare_next_inner_iteration(self, iteration: int) -> None:
        """
        Method to prepare the internal state for the next inner iteration of the current timestep

        :param iteration: Number of the next inner iteration to prepare for
        """
    def calculate_gravity_force(self, density):

        result = (((self.valve_density - density)
                   / self.valve_density)
                  * (self.valve_mass + self.spring_mass) * self._g)
        return result

    def calculate_spring_force(self, displacement):
        """
        Method to calculate the reacting force depending on the displacement of the valve

        :param displacement: Displacement of the valve
        :return: Returns the spring force
        """
        result = self._spring_force0 + self._spring_stiffness * displacement

        return result

    def calculate_lower_pressure_force(self, pressure):
        """


        :param pressure:
        :return:
        """

        result = (self.valve_area - self.contact_area) * pressure

        return result

    def calculate_upper_pressure_force(self, pressure):

        result = self.valve_area * pressure

        return result

    def calculate_contact_pressure(self, lower_pressure, upper_pressure, displacement, velocity, density, viscosity):

        if displacement <= 1e-7:

            pk = (lower_pressure * ((self._outer_radius / self.radius - 1)
                                    / (self._outer_radius / self._inner_radius - 1))
                  + upper_pressure * ((1 - self._inner_radius / self.radius)
                                      / (1 - self._inner_radius / self._outer_radius)))

        elif self.flow_counter:
            pk = (lower_pressure * ((self._outer_radius / self.radius - 1)
                                    / (self._outer_radius / self._inner_radius - 1))
                  + upper_pressure * ((1 - self._inner_radius / self.radius)
                                      / (1 - self._inner_radius / self._outer_radius)))

        else:
            pk = (lower_pressure * ((self._outer_radius / self.radius - 1)
                                    / (self._outer_radius / self._inner_radius - 1))
                  + upper_pressure * ((1 - self._inner_radius / self.radius)
                                      / (1 - self._inner_radius / self._outer_radius))
                  + (6.0 * velocity * viscosity * density
                     * (self._outer_radius**2 - self._inner_radius**2)
                     * (self.radius - self._inner_radius)
                     * (self._outer_radius - self.radius))
                  / ((displacement**3) * ((np.sin(self._seat_tilt))**2)
                     * self.radius * (self._outer_radius - self._inner_radius))
                  )

        for i in range(pk.shape[0]):
            if pk[i] <= 2300:
                pk[i] = 2300

        f1 = pk * self.radius * np.pi

        result = integrate.simps(f1, self.radius)

        return result

    def calculate_dampening_force(self, density, velocity, viscosity):

        teta = density * np.abs(velocity) * self._flow_constant_1 + self._flow_constant_2 * viscosity * density

        result = teta * velocity

        return result

    def calculate_gap_reynolds_number(self, displacement, volume_flow, viscosity):

        averaged_diameter = (self._outer_diameter + self._inner_diameter) / 2.0
        averaged_gap_diameter = averaged_diameter - (displacement / 2.0) * np.sin(2.0 * self._seat_tilt)
        self._gap_area[1, 0] = averaged_gap_diameter * np.pi * displacement * np.sin(self._seat_tilt)
        gap_reynolds_number = 2.0 * np.abs(volume_flow - (averaged_diameter**2)/4.0
                                           * np.pi * self._valve_velocity[1, 0]) / (viscosity
                                                                                    * np.pi
                                                                                    * (self._gap_area[1, 0]))

        return gap_reynolds_number

    def calculate_dimensionless_coefficient_of_force(self, reynolds_number, displacement):
        if reynolds_number == 0.0:
            result = 0.5
        else:
            result = (self._factor_k0
                      - self._factor_k1 * (displacement / self._outer_diameter)
                      - self._factor_k2 * np.log(reynolds_number))

        return result

    def calculate_flow_force(self, dimensionless_coefficient, density, lower_pressure, upper_pressure):

        result = dimensionless_coefficient * (lower_pressure - upper_pressure) * (self.outer_diameter**2) * np.pi / 4.0
        return result

    def calculate_zeta_value(self, displacement, reynolds_number):

        if reynolds_number > 0.0:
            result = ((self._friction_factor_a / reynolds_number + self._friction_factor_c)
                      * (self._friction_factor_b
                         * np.power(np.abs(self.inner_diameter / displacement), self._friction_factor_d) + 1))
        else:
            result = self._valve_zeta[2, 0]
        return result

    def calculate_flow(self):

        pressure_a = self._pressure[1, 0]
        velocity_a = self._velocity[1, 0]
        friction_a = self._friction[1, 0]
        pressure_b = self._pressure[1, -1]
        velocity_b = self._velocity[1, -1]
        friction_b = self._friction[1, -1]
        area_a = self._area[0]
        area_b = self._area[1]
        sos_a = self._sos[1, 0]
        sos_b = self._sos[1, -1]

        density_a = self.fluid.density(pressure=pressure_a, temperature=None)
        density_b = self.fluid.density(pressure=pressure_b, temperature=None)
        viscosity = self.fluid.viscosity(temperature=None, shear_rate=None) / density_a

        f1 = (density_a * sos_a * velocity_a
              + pressure_a
              - friction_a * self._delta_t * density_a * sos_a)

        f2 = (- density_b * sos_b * velocity_b
              + pressure_b
              + friction_b * self._delta_t * density_b * sos_b)

        k = (self._gap_area[1, 0] * self._gap_area[1, 0]) / (self._valve_zeta[1, 0] * density_a)

        b = (k * (density_a * sos_a / area_a
                  + density_b * sos_b / area_b))

        c = f1 - f2

        self._volume_flow[0, 1] = np.sign(c) * (-b + np.sqrt(b * b + np.sign(c) * 2 * k * c))
        self._volume_flow[0, 1] = self._volume_flow[0, 1] + self.valve_area * self._valve_velocity[0, 0]

        self._pressure[0, 1] = (density_a * sos_a * velocity_a
                                - density_a * sos_a * (self._volume_flow[0, 1]/area_a)
                                + pressure_a
                                - friction_a * self._delta_t * density_a * sos_a)

        self._pressure[0, -2] = (- density_b * sos_b * velocity_b
                                 + density_b * sos_b * (self._volume_flow[0, 1] / area_b)
                                 + pressure_b
                                 + friction_b * self._delta_t * density_b * sos_b)

        self._lower_pressure[0, 0] = self._pressure[0, 1]
        self._upper_pressure[0, 0] = self._pressure[0, -2]
        self._delta_p[0, 0] = self._lower_pressure[0, 0] - self._upper_pressure[0, 0]

        return None

    def calculate_valve(self, lower_pressure, upper_pressure, volume_flow, viscosity, density):

        displacement = self._valve_displacement[1, 0]
        velocity = self._valve_velocity[1, 0]
        epsilon = 1e-6

        # Check: Is valve closed?

        if displacement <= 0:

            # Valve is closed!
            self.flow_counter = False
            # Calculate forces of the closed regime!

            self._spring_force[1, 0] = self.calculate_spring_force(displacement)
            self._gravity_force[1, 0] = self.calculate_gravity_force(density)
            self._upper_pressure_force[1, 0] = self.calculate_upper_pressure_force(upper_pressure)
            self._lower_pressure_force[1, 0] = self.calculate_lower_pressure_force(lower_pressure)
            self._contact_pressure_force[1, 0] = self.calculate_contact_pressure(lower_pressure,
                                                                                 upper_pressure,
                                                                                 0.0,
                                                                                 0.0,
                                                                                 density,
                                                                                 viscosity)

            # Check: Are upper forces bigger than lower forces?

            if self._gravity_force[1, 0] + self._spring_force[1, 0] + self._upper_pressure_force[1, 0]\
                    <= self._lower_pressure_force[1, 0] + self._contact_pressure_force[1, 0]:

                # Lower forces are bigger than upper forces --> Valve moves!

                self._damping_force[1, 0] = self.calculate_dampening_force(density, velocity, viscosity)

                # Calculate forces for wall close regime calculation

                forces = (self._lower_pressure_force[1, 0]
                          + self._contact_pressure_force[1, 0]
                          - self._gravity_force[1, 0]
                          - self._spring_force[1, 0]
                          - self._damping_force[1, 0]
                          - self._upper_pressure_force[1, 0])

                # Result is the current displacement of the valve

                result = (forces * (self._delta_t**2)
                          / (self._valve_mass + self.spring_mass * (1 / 3.0))
                          + 2.0 * displacement - self._valve_displacement[2, 0])

                if result >= self.max_displacement:

                    result = self.max_displacement

                else:

                    result = result

                # Calculating results

                self._valve_displacement[0, 0] = result
                self._valve_velocity[0, 0] = (result - displacement) / self._delta_t
                self._valve_acceleration[0, 0] = ((result - 2.0 * displacement + self._valve_displacement[2, 0])
                                                  / (self._delta_t**2))

                # For close Wall opening the valve has no flow through it

                no_flow = True

                self._cases.append('Valve Closed and starts to open')

                return no_flow

            # Check: Are upper forces bigger than lower forces?

            else:

                # Upper forces are bigger than lower forces --> Valve stays closed
                # Result is 0.0 --> no smaller displacement than 0.0 is allowed!

                result = 0.0
                self._valve_displacement[0, 0] = result
                self._valve_velocity[0, 0] = (result - displacement) / self._delta_t
                self._valve_acceleration[0, 0] = ((result - 2.0 * displacement + self._valve_displacement[2, 0])
                                                  / (self._delta_t**2))

                # Valve closed now flow through it
                self._cases.append('Valve Closed and stays closed')

                no_flow = True

                return no_flow

        # Check: Is Valve closed?

        else:

            # Valve is OPEN! --> displacement > 0.0!

            contact_pressure_force0 = self.calculate_contact_pressure(lower_pressure,
                                                                      upper_pressure,
                                                                      displacement,
                                                                      0.0,
                                                                      density,
                                                                      viscosity)

            self._contact_pressure_force[1, 0] = self.calculate_contact_pressure(lower_pressure,
                                                                                 upper_pressure,
                                                                                 displacement,
                                                                                 velocity,
                                                                                 density,
                                                                                 viscosity)

            self._spring_force[1, 0] = self.calculate_spring_force(displacement)
            self._gravity_force[1, 0] = self.calculate_gravity_force(density)
            self._upper_pressure_force[1, 0] = self.calculate_upper_pressure_force(upper_pressure)
            self._lower_pressure_force[1, 0] = self.calculate_lower_pressure_force(lower_pressure)
            self._damping_force[1, 0] = self.calculate_dampening_force(density, velocity, viscosity)

            # Check: Close to wall regime?
            self.epsilon.append(np.abs(np.abs(self._contact_pressure_force[1, 0]) - np.abs(contact_pressure_force0)) / np.abs(contact_pressure_force0))

            if np.abs(np.abs(self._contact_pressure_force[1, 0]) - np.abs(contact_pressure_force0)) / np.abs(contact_pressure_force0)\
                    >= epsilon:

                # Close to wall!
                # Calculate forces for wall close regime calculation

                forces = (self._lower_pressure_force[1, 0]
                          + self._contact_pressure_force[1, 0]
                          - self._gravity_force[1, 0]
                          - self._spring_force[1, 0]
                          - self._damping_force[1, 0]
                          - self._upper_pressure_force[1, 0])

                # Result is the current displacement of the valve

                result = (forces * (self._delta_t**2)
                          / (self._valve_mass + self.spring_mass * (1 / 3.0))
                          + 2.0 * displacement - self._valve_displacement[2, 0])

                # Check: Is the displacement > 0.0?

                if result <= 0.0:

                    # Valve is closed!

                    result = 0.0

                elif result >= self.max_displacement:

                    result = self.max_displacement

                    # Valve is still open
                else:

                    result = result

                # Calculating results
                self._valve_displacement[0, 0] = result
                self._valve_velocity[0, 0] = (result - displacement) / self._delta_t
                self._valve_acceleration[0, 0] = ((result - 2.0 * displacement + self._valve_displacement[2, 0])
                                                  / (self._delta_t**2))

                # For close Wall opening the valve has no flow through it

                self._cases.append('Close Wall regime')

                no_flow = True

                return no_flow

            # Check: Close to wall regime?

            else:

                # No! --> Flow regime!
                self.flow_counter = True

                reynolds_number = self.calculate_gap_reynolds_number(displacement,
                                                                     volume_flow,
                                                                     viscosity)

                flow_coefficient = self.calculate_dimensionless_coefficient_of_force(reynolds_number,
                                                                                     displacement)

                self._valve_zeta[1, 0] = self.calculate_zeta_value(displacement, reynolds_number)

                self._flow_force[1, 0] = self.calculate_flow_force(flow_coefficient,
                                                                   density,
                                                                   lower_pressure,
                                                                   upper_pressure)

                forces = (self._flow_force[1, 0]
                          - self._gravity_force[1, 0]
                          - self._spring_force[1, 0]
                          - self._damping_force[1, 0]
                          )

                # Flow Regime! --> Volumeflow through Valve allowed!

                no_flow = False

                result = (forces * (self._delta_t**2)
                          / (self._valve_mass + self.spring_mass * (1 / 3.0))
                          + 2.0 * displacement - self._valve_displacement[2, 0])

                # Check: Is Valve Closed?

                if result <= 0.0:

                    # Valve Closed!

                    self.flow_counter = False

                    result = 0.0

                # Check: Is Valve at maximum displacement?

                elif result >= self.max_displacement:

                    # Valve is at maximum allowed displacement

                    result = self.max_displacement

                # Valve is somewhere else!

                else:
                    result = result

                self._valve_displacement[0, 0] = result
                self._valve_velocity[0, 0] = (result - displacement) / self._delta_t
                self._valve_acceleration[0, 0] = ((result - 2.0 * displacement + self._valve_displacement[2, 0])
                                                  / (self._delta_t**2))
                self.calculate_flow()

                self._cases.append('Flow regime')

                return no_flow

    def calculate_next_inner_iteration(self, iteration: int) -> bool:
        """
        Method to do the calculations of the next inner iteration

        :param iteration: Number of the next inner iteration
        :return: Whether this component needs another inner iteration afterwards
        """
        # Get the input fields
        pressure_a = self._pressure[1, 0]
        velocity_a = self._velocity[1, 0]
        friction_a = self._friction[1, 0]
        pressure_b = self._pressure[1, -1]
        velocity_b = self._velocity[1, -1]
        friction_b = self._friction[1, -1]
        area_a = self._area[0]
        area_b = self._area[1]
        sos_a = self._sos[1, 0]
        sos_b = self._sos[1, -1]

        # Calculate fluid properties
        density_a = self.fluid.density(pressure=pressure_a, temperature=None)
        density_b = self.fluid.density(pressure=pressure_b, temperature=None)
        viscosity = self.fluid.viscosity(temperature=None, shear_rate=None) / density_a

        no_flow = self.calculate_valve(self._lower_pressure[1, 0], self._upper_pressure[1, 0],
                                       self._volume_flow[1, 1], viscosity, density_a)

        if no_flow:
            self._volume_flow[0, 1] = self.valve_area * self._valve_velocity[0, 0]

            self._pressure[0, 1] = (density_a * sos_a * velocity_a
                                    - density_a * sos_a * (self._volume_flow[0, 1] / area_a)
                                    + pressure_a - friction_a * self._delta_t * density_a * sos_a)

            self._pressure[0, -2] = (- density_b * sos_b * velocity_b
                                     + density_b * sos_b * (self._volume_flow[0, 1] / area_b)
                                     + pressure_b + friction_b * self._delta_t * density_b * sos_b)

            self._lower_pressure[0, 0] = self._pressure[0, 1]
            self._upper_pressure[0, 0] = self._pressure[0, -2]
            self._delta_p[0, 0] = self._lower_pressure[0, 0] - self._upper_pressure[0, 0]

        return False
