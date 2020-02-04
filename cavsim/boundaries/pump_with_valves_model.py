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
            self,                                       # -
            suction_valve_density: float,               # kg/m³
            suction_spring_force0: float,               # N
            suction_spring_stiffness: float,            # N/m
            suction_spring_mass: float,                 # kg
            suction_valve_mass: float,                  # kg
            suction_outer_diameter: float,              # m
            suction_inner_contact_diameter: float,      # m
            suction_seat_tilt: float,                   # °
            suction_flow_constant_1: float,             # -
            suction_flow_constant_2: float,             # -
            suction_friction_factor_a: float,           # -
            suction_friction_factor_b: float,           # -
            suction_friction_factor_c: float,           # -
            suction_friction_factor_d: float,           # -
            suction_factor_k0: float,                   # -
            suction_factor_k1: float,                   # -
            suction_factor_k2: float,                   # -
            suction_max_displacement,                   # m
            suction_outer_contact_diameter: float,      # m
            discharge_valve_density: float,             # kg/m³
            discharge_spring_force0: float,             # N
            discharge_spring_stiffness: float,          # N/m
            discharge_spring_mass: float,               # kg
            discharge_valve_mass: float,                # kg
            discharge_outer_diameter: float,            # m
            discharge_inner_contact_diameter: float,    # m
            discharge_seat_tilt: float,                 # °
            discharge_flow_constant_1: float,           # -
            discharge_flow_constant_2: float,           # -
            discharge_friction_factor_a: float,         # -
            discharge_friction_factor_b: float,         # -
            discharge_friction_factor_c: float,         # -
            discharge_friction_factor_d: float,         # -
            discharge_factor_k0: float,                 # -
            discharge_factor_k1: float,                 # -
            discharge_factor_k2: float,                 # -
            discharge_max_displacement,                 # m
            discharge_outer_contact_diameter: float,    # m
            pump_radius: float,                         # m
            rpm: float,                                 # U/min
            rratio: float,                              # -
            phi0: float,                                # °
            piston_diameter: float,                     # m
            death_volume: float,                        # m³

    ) -> None:

        super(PumpSuctionValve, self).__init__()
        # Register values
        # Suction Valve values
        self._suction_valve_density = suction_valve_density
        self._suction_spring_force0 = suction_spring_force0
        self._suction_spring_stiffness = suction_spring_stiffness
        self._suction_outer_diameter = suction_outer_diameter
        self._suction_inner_contact_diameter = suction_inner_contact_diameter
        self._suction_seat_tilt = suction_seat_tilt
        self._suction_valve_mass = suction_valve_mass
        self._suction_spring_mass = suction_spring_mass
        self._suction_flow_constant_1 = suction_flow_constant_1
        self._suction_flow_constant_2 = suction_flow_constant_2
        self._suction_friction_factor_a = suction_friction_factor_a
        self._suction_friction_factor_b = suction_friction_factor_b
        self._suction_friction_factor_c = suction_friction_factor_c
        self._suction_friction_factor_d = suction_friction_factor_d
        self._suction_factor_k0 = suction_factor_k0
        self._suction_factor_k1 = suction_factor_k1
        self._suction_factor_k2 = suction_factor_k2
        self._suction_max_displacement = suction_max_displacement
        self._suction_outer_contact_diameter = suction_outer_contact_diameter

        # Discharge Valve values
        self._discharge_valve_density = discharge_valve_density
        self._discharge_spring_force0 = discharge_spring_force0
        self._discharge_spring_stiffness = discharge_spring_stiffness
        self._discharge_outer_diameter = discharge_outer_diameter
        self._discharge_inner_contact_diameter = discharge_inner_contact_diameter
        self._discharge_seat_tilt = discharge_seat_tilt
        self._discharge_valve_mass = discharge_valve_mass
        self._discharge_spring_mass = discharge_spring_mass
        self._discharge_flow_constant_1 = discharge_flow_constant_1
        self._discharge_flow_constant_2 = discharge_flow_constant_2
        self._discharge_friction_factor_a = discharge_friction_factor_a
        self._discharge_friction_factor_b = discharge_friction_factor_b
        self._discharge_friction_factor_c = discharge_friction_factor_c
        self._discharge_friction_factor_d = discharge_friction_factor_d
        self._g = 9.81
        self._discharge_factor_k0 = discharge_factor_k0
        self._discharge_factor_k1 = discharge_factor_k1
        self._discharge_factor_k2 = discharge_factor_k2
        self._discharge_max_displacement = discharge_max_displacement
        self._discharge_outer_contact_diameter = discharge_outer_contact_diameter

        # Pump values
        self.pump_radius = pump_radius
        self._pump_rpm = rpm
        self._rratio = rratio
        self.phi0 = phi0 * np.pi / 180.0
        self._piston_diameter = piston_diameter
        self.stroke = 0.0
        self.piston_velocity = 0.0
        self._pump_death_volume = death_volume

        self._cases = []
        self.flow_counter = False
        self.epsilon = []
        self.epsilon.append(0.0)
        self.time = 0.0

        # Register internal fields

        self._pressure: np.ndarray = self.field_create('pressure', 3)
        self._suction_upper_pressure: np.ndarray = self.field_create('upper_pressure', 3)
        self._suction_lower_pressure: np.ndarray = self.field_create('lower_pressure', 3)


        self._suction_valve_displacement = self.field_create('suction_displacement', 4)
        self._suction_valve_velocity = self.field_create('suction_valve_velocity', 4)
        self._suction_valve_acceleration = self.field_create('suction_acceleration', 4)
        self._suction_spring_force: np.ndarray = self.field_create('suction_spring_force', 3)
        self._suction_gravity_force: np.ndarray = self.field_create('suction_gravity_force', 3)
        self._suction_acceleration_force: np.ndarray = self.field_create('suction_acceleration_force', 3)
        self._suction_damping_force: np.ndarray = self.field_create('suction_damping_force', 3)
        self._suction_upper_pressure_force: np.ndarray = self.field_create('suction_upper_pressure_force', 3)
        self._suction_lower_pressure_force: np.ndarray = self.field_create('suction_lower_pressure_force', 3)
        self._suction_contact_pressure_force: np.ndarray = self.field_create('suction_contact_pressure_force', 3)
        self._suction_flow_force: np.ndarray = self.field_create('suction_flow_force', 3)
        self._suction_valve_zeta = self.field_create('suction_valve_zeta', 3)
        self._suction_gap_area = self.field_create('suction_gap_area', 3)
        self._suction_delta_p = self.field_create('suction_delta_p', 3)
        self.mass_flow_suction_valve: np.ndarray = self.field_create('mass flow suction valve', 3)

        self._discharge_upper_pressure: np.ndarray = self.field_create('upper_pressure', 3)
        self._discharge_lower_pressure: np.ndarray = self.field_create('lower_pressure', 3)
        self._discharge_valve_displacement = self.field_create('displacement', 4)
        self._discharge_valve_velocity = self.field_create('valve_velocity', 4)
        self._discharge_valve_acceleration = self.field_create('acceleration', 4)
        self._discharge_spring_force: np.ndarray = self.field_create('spring_force', 3)
        self._discharge_gravity_force: np.ndarray = self.field_create('gravity_force', 3)
        self._discharge_acceleration_force: np.ndarray = self.field_create('acceleration_force', 3)
        self._discharge_damping_force: np.ndarray = self.field_create('damping_force', 3)
        self._discharge_upper_pressure_force: np.ndarray = self.field_create('upper_pressure_force', 3)
        self._discharge_lower_pressure_force: np.ndarray = self.field_create('lower_pressure_force', 3)
        self._discharge_contact_pressure_force: np.ndarray = self.field_create('contact_pressure_force', 3)
        self._discharge_flow_force: np.ndarray = self.field_create('flow_force', 3)
        self._discharge_valve_zeta = self.field_create('valve_zeta', 3)
        self._discharge_gap_area = self.field_create('gap_area', 3)
        self._discharge_delta_p = self.field_create('delta_p', 3)
        self.mass_flow_discharge_valve: np.ndarray = self.field_create('mass flow discharge valve', 3)
        self._discharge_volume_flow: np.ndarray = self.field_create('discharge volume flow', 3)

        self._velocity: np.ndarray = self.field_create('velocity', 3)
        self._volume_flow: np.ndarray = self.field_create('volume_flow', 3)
        self._sos: np.ndarray = self.field_create('speed_of_sound', 3)
        self._friction = self.field_create('friction', 3)
        self._area = np.empty(2)

        # Register Pump Fields
        self._pump_volume_change: np.ndarray = self.field_create('pump_volume_change', 3)
        self._pump_volume: np.ndarray = self.field_create('pump_volume', 3)
        self.pump_pressure: np.ndarray = self.field_create('pump_pressure', 3)



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
    def suction_valve_density(self):

        return self._suction_valve_density

    @property
    def suction_spring_stiffness(self):

        return self._suction_spring_stiffness

    @property
    def suction_spring_force0(self):

        return self._suction_spring_force0

    @property
    def suction_valve_mass(self):

        return self._suction_valve_mass

    @property
    def suction_spring_mass(self):

        return self._suction_spring_mass

    @property
    def suction_outer_diameter(self):

        return self._suction_outer_diameter

    @property
    def suction_inner_contact_diameter(self):

        return self._suction_inner_contact_diameter

    @property
    def suction_outer_contact_diameter(self):

        return self._suction_outer_contact_diameter

    @property
    def suction_valve_area(self):

        return np.pi * np.power(self.suction_outer_diameter, 2) / 4

    @property
    def suction_radius(self):

        return np.linspace(self.suction_inner_contact_radius, self.suction_outer_contact_radius, 1000)

    @property
    def suction_inner_contact_radius(self):

        return self.suction_inner_contact_diameter / 2.0

    @property
    def suction_outer_contact_radius(self):

        return self.suction_outer_contact_diameter / 2.0

    @property
    def suction_contact_area(self):

        return np.pi * ((self.suction_outer_contact_diameter / 2)**2 - (self.suction_inner_contact_diameter / 2)**2)

    @property
    def suction_flow_constant_1(self):

        return self._suction_flow_constant_1

    @property
    def suction_flow_constant_2(self):

        return self._suction_flow_constant_2

    @property
    def suction_friction_factor_a(self):

        return self._suction_friction_factor_a

    @property
    def suction_friction_factor_b(self):

        return self._suction_friction_factor_b

    @property
    def suction_friction_factor_c(self):

        return self._suction_friction_factor_c

    @property
    def suction_friction_factor_d(self):

        return self._suction_friction_factor_d

    @property
    def suction_factor_k0(self):

        return self._suction_factor_k0

    @property
    def suction_factor_k1(self):

        return self._suction_factor_k1

    @property
    def suction_factor_k2(self):

        return self._suction_factor_k2

    @property
    def suction_max_displacement(self):

        return self._suction_max_displacement











    @property
    def discharge_valve_density(self):

        return self._discharge_valve_density

    @property
    def discharge_spring_stiffness(self):

        return self._discharge_spring_stiffness

    @property
    def discharge_spring_force0(self):

        return self._discharge_spring_force0

    @property
    def discharge_valve_mass(self):

        return self._discharge_valve_mass

    @property
    def discharge_spring_mass(self):

        return self._discharge_spring_mass

    @property
    def discharge_outer_diameter(self):

        return self._discharge_outer_diameter

    @property
    def discharge_inner_contact_diameter(self):

        return self._discharge_inner_contact_diameter

    @property
    def discharge_outer_contact_diameter(self):

        return self._discharge_outer_contact_diameter

    @property
    def discharge_valve_area(self):

        return np.pi * np.power(self.discharge_outer_diameter, 2) / 4

    @property
    def discharge_radius(self):

        return np.linspace(self.discharge_inner_contact_radius, self.discharge_outer_contact_radius, 10000)

    @property
    def discharge_inner_contact_radius(self):

        return self.discharge_inner_contact_diameter / 2.0

    @property
    def discharge_outer_contact_radius(self):

        return self.discharge_outer_contact_diameter / 2.0

    @property
    def discharge_contact_area(self):

        return np.pi * ((self.discharge_outer_contact_diameter / 2)**2 - (self.discharge_inner_contact_diameter / 2)**2)

    @property
    def discharge_flow_constant_1(self):

        return self._discharge_flow_constant_1

    @property
    def discharge_flow_constant_2(self):

        return self._discharge_flow_constant_2

    @property
    def discharge_friction_factor_a(self):

        return self._discharge_friction_factor_a

    @property
    def discharge_friction_factor_b(self):

        return self._discharge_friction_factor_b

    @property
    def discharge_friction_factor_c(self):

        return self._discharge_friction_factor_c

    @property
    def discharge_friction_factor_d(self):

        return self._discharge_friction_factor_d

    @property
    def discharge_factor_k0(self):

        return self._discharge_factor_k0

    @property
    def discharge_factor_k1(self):

        return self._discharge_factor_k1

    @property
    def discharge_factor_k2(self):

        return self._discharge_factor_k2

    @property
    def discharge_max_displacement(self):

        return self._discharge_max_displacement

    @property
    def omega(self):

        return self.pump_rpm * 2.0 * np.pi / 60.0

    @property
    def pump_rpm(self):

        return self._pump_rpm

    @property
    def piston_diameter(self):

        return self._piston_diameter

    @property
    def piston_area(self):

        return self.piston_diameter**2 / 4.0 * np.pi

    @property
    def rratio(self):

        return self._rratio

    @property
    def pump_death_volume(self):

        return self._pump_death_volume












































    def calculate_piston_stroke(self, time):

        self.stroke = (self.pump_radius *
                       (1 - np.cos(self.omega * time + self.phi0)
                        + self.rratio / 2.0 * (np.power(np.sin(self.omega * time), 2))))
        return None

    def calculate_piston_velocity(self, time):

        self.piston_velocity = (self.pump_radius * (self.omega * time + self.phi0)
                                * (np.sin(self.omega * time + self.phi0)
                                   + self.rratio / 2.0 * np.sin(2 * (self.omega * time + self.phi0))))

        return None

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
        self.field('spring_force')[:, :] = np.zeros(self.field('spring_force').shape)[:, :] * self._suction_spring_force0
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
        # Set new total time
        self.time = +self._delta_t

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
    def calculate_suction_gravity_force(self, density):

        result = (((self.suction_valve_density - density)
                   / self.suction_valve_density)
                  * (self.suction_valve_mass + self.suction_spring_mass) * self._g)
        return result

    def calculate_suction_spring_force(self, displacement):
        """
        Method to calculate the reacting force depending on the displacement of the valve

        :param displacement: Displacement of the valve
        :return: Returns the spring force
        """
        result = self.suction_spring_force0 + self._suction_spring_stiffness * displacement

        return result

    def calculate_suction_lower_pressure_force(self, pressure):
        """


        :param pressure:
        :return:
        """

        result = self.suction_contact_area * pressure

        return result

    def calculate_suction_upper_pressure_force(self, pressure):

        result = self.suction_valve_area * pressure

        return result

    def calculate_suction_contact_pressure(self, lower_pressure, upper_pressure, displacement, velocity, density, viscosity):

        if displacement <= 1e-7:

            pk = (lower_pressure * ((self.suction_outer_contact_radius / self.suction_radius - 1)
                                    / (self.suction_outer_contact_radius / self.suction_inner_contact_radius - 1))
                  + upper_pressure * ((1 - self.suction_inner_contact_radius / self.suction_radius)
                                      / (1 - self.suction_inner_contact_radius / self.suction_outer_contact_radius)))

        elif self.flow_counter:
            pk = (lower_pressure * ((self.suction_outer_contact_radius / self.suction_radius - 1)
                                    / (self.suction_outer_contact_radius / self.suction_inner_contact_radius - 1))
                  + upper_pressure * ((1 - self.suction_inner_contact_radius / self.suction_radius)
                                      / (1 - self.suction_inner_contact_radius / self.suction_outer_contact_radius)))

        else:
            pk = (lower_pressure * ((self.suction_outer_contact_radius / self.suction_radius - 1)
                                    / (self.suction_outer_contact_radius / self.suction_inner_contact_radius - 1))
                  + upper_pressure * ((1 - self.suction_inner_contact_radius / self.suction_radius)
                                      / (1 - self.suction_inner_contact_radius / self.suction_outer_contact_radius))
                  + (6.0 * velocity * viscosity * density
                     * (self.suction_outer_contact_radius**2 - self.suction_inner_contact_radius**2)
                     * (self.suction_radius - self.suction_inner_contact_radius)
                     * (self.suction_outer_contact_radius - self.suction_radius))
                  / ((displacement**3) * ((np.sin(self._suction_seat_tilt))**2)
                     * self.suction_radius * (self.suction_outer_contact_radius - self.suction_inner_contact_radius))
                  )

        for i in range(pk.shape[0]):
            if pk[i] <= 2300:
                pk[i] = 2300

        f1 = pk * self.suction_radius * np.pi

        result = integrate.simps(f1, self.suction_radius)

        return result

    def calculate_suction_dampening_force(self, density, velocity, viscosity):

        teta = density * np.abs(velocity) * self._suction_flow_constant_1\
               + self._suction_flow_constant_2 * viscosity * density

        result = teta * velocity

        return result

    def calculate_suction_gap_reynolds_number(self, displacement, volume_flow, viscosity):

        averaged_diameter = (self.suction_outer_diameter + self.suction_inner_contact_diameter) / 2.0
        averaged_gap_diameter = averaged_diameter - (displacement / 2.0) * np.sin(2.0 * self._suction_seat_tilt)
        self._suction_gap_area[1, 0] = averaged_gap_diameter * np.pi * displacement * np.sin(self._suction_seat_tilt)
        gap_reynolds_number = 2.0 * np.abs(volume_flow - (averaged_diameter**2)/4.0
                                           * np.pi * self._suction_valve_velocity[1, 0]) / (viscosity
                                                                                            * np.pi
                                                                                            * (self._suction_gap_area[1, 0]))

        return gap_reynolds_number

    def calculate_suction_dimensionless_coefficient_of_force(self, reynolds_number, displacement):
        if reynolds_number == 0.0:
            result = 0.5
        else:
            result = (self._suction_factor_k0
                      - self._suction_factor_k1 * (displacement / self.suction_outer_diameter)
                      - self._suction_factor_k2 * np.log(reynolds_number))

        return result

    def calculate_suction_flow_force(self, dimensionless_coefficient, density, lower_pressure, upper_pressure):

        result = dimensionless_coefficient * (lower_pressure - upper_pressure) * (self.suction_outer_diameter**2) * np.pi / 4.0
        return result

    def calculate_suction_zeta_value(self, displacement, reynolds_number):

        if reynolds_number > 0.0:
            result = ((self._suction_friction_factor_a / reynolds_number + self._suction_friction_factor_c)
                      * (self._suction_friction_factor_b
                         * np.power(np.abs(self.suction_inner_contact_diameter / displacement), self._suction_friction_factor_d) + 1))
        else:
            result = self._suction_valve_zeta[2, 0]
        return result

    def calculate_suction_flow(self, no_flow):

        pressure_a = self._pressure[1, 0]
        velocity_a = self._velocity[1, 0]
        friction_a = self._friction[1, 0]
        area_a = self._area[0]
        sos_a = self._sos[1, 0]
        density_a = self.fluid.density(pressure=pressure_a, temperature=None)
        if no_flow:
            self._volume_flow[0, 0] = 0.0 + self.suction_valve_area * self._suction_valve_velocity[0, 0]
        else:

            self._volume_flow[0, 0] = (self.mass_flow_suction_valve[0, 0] / density_a
                                       + self.suction_valve_area * self._suction_valve_velocity[0, 0])

        velocity = self._volume_flow[0, 0] / area_a

        self._pressure[0, 1] = (density_a * sos_a * velocity_a
                                - density_a * sos_a * velocity
                                + pressure_a
                                - friction_a * self._delta_t * density_a * sos_a)

        return None

    def calculate_suction_valve(self, lower_pressure, upper_pressure, viscosity, density):

        displacement = self._suction_valve_displacement[1, 0]
        velocity = self._suction_valve_velocity[1, 0]
        epsilon = 1e-6
        volume_flow = self.mass_flow_suction_valve[1, 0]

        # Check: Is valve closed?

        if displacement <= 0:

            # Valve is closed!
            self.flow_counter = False
            # Calculate forces of the closed regime!

            self._suction_spring_force[1, 0] = self.calculate_suction_spring_force(displacement)
            self._suction_gravity_force[1, 0] = self.calculate_suction_gravity_force(density)
            self._suction_upper_pressure_force[1, 0] = self.calculate_suction_upper_pressure_force(upper_pressure)
            self._suction_lower_pressure_force[1, 0] = self.calculate_suction_lower_pressure_force(lower_pressure)
            self._suction_contact_pressure_force[1, 0] = self.calculate_suction_contact_pressure(lower_pressure,
                                                                                                 upper_pressure,
                                                                                                 0.0,
                                                                                                 0.0,
                                                                                                 density,
                                                                                                 viscosity)

            # Check: Are upper forces bigger than lower forces?

            if self._suction_gravity_force[1, 0] + self._suction_spring_force[1, 0] + self._suction_upper_pressure_force[1, 0]\
                    <= self._suction_lower_pressure_force[1, 0] + self._suction_contact_pressure_force[1, 0]:

                # Lower forces are bigger than upper forces --> Valve moves!

                self._suction_damping_force[1, 0] = self.calculate_suction_dampening_force(density, velocity, viscosity)

                # Calculate forces for wall close regime calculation

                forces = (self._suction_lower_pressure_force[1, 0]
                          + self._suction_contact_pressure_force[1, 0]
                          - self._suction_gravity_force[1, 0]
                          - self._suction_spring_force[1, 0]
                          - self._suction_damping_force[1, 0]
                          - self._suction_upper_pressure_force[1, 0])

                # Result is the current displacement of the valve

                result = (forces * (self._delta_t**2)
                          / (self._suction_valve_mass + self.suction_spring_mass * (1 / 3.0))
                          + 2.0 * displacement - self._suction_valve_displacement[2, 0])

                if result >= self.suction_max_displacement:

                    result = self.suction_max_displacement
                    self.mass_flow_suction_valve[0, 0] = 0.0

                else:

                    result = result
                    self.mass_flow_suction_valve[0, 0] = 0.0

                # Calculating results

                self._suction_valve_displacement[0, 0] = result
                self._suction_valve_velocity[0, 0] = (result - displacement) / self._delta_t
                self._suction_valve_acceleration[0, 0] = ((result - 2.0 * displacement + self._suction_valve_displacement[2, 0])
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
                self._suction_valve_displacement[0, 0] = result
                self._suction_valve_velocity[0, 0] = (result - displacement) / self._delta_t
                self._suction_valve_acceleration[0, 0] = ((result - 2.0 * displacement + self._suction_valve_displacement[2, 0])
                                                          / (self._delta_t**2))
                self.mass_flow_suction_valve[0, 0] = 0.0

                # Valve closed now flow through it
                self._cases.append('Valve Closed and stays closed')

                no_flow = True

                return no_flow

        # Check: Is Valve closed?

        else:

            # Valve is OPEN! --> displacement > 0.0!

            contact_pressure_force0 = self.calculate_suction_contact_pressure(lower_pressure,
                                                                              upper_pressure,
                                                                              displacement,
                                                                              0.0,
                                                                              density,
                                                                              viscosity)

            self._suction_contact_pressure_force[1, 0] = self.calculate_suction_contact_pressure(lower_pressure,
                                                                                                 upper_pressure,
                                                                                                 displacement,
                                                                                                 velocity,
                                                                                                 density,
                                                                                                 viscosity)

            self._suction_spring_force[1, 0] = self.calculate_suction_spring_force(displacement)
            self._suction_gravity_force[1, 0] = self.calculate_suction_gravity_force(density)
            self._suction_upper_pressure_force[1, 0] = self.calculate_suction_upper_pressure_force(upper_pressure)
            self._suction_lower_pressure_force[1, 0] = self.calculate_suction_lower_pressure_force(lower_pressure)
            self._suction_damping_force[1, 0] = self.calculate_suction_dampening_force(density, velocity, viscosity)

            # Check: Close to wall regime?
            self.epsilon.append(np.abs(np.abs(self._suction_contact_pressure_force[1, 0]) - np.abs(contact_pressure_force0)) / np.abs(contact_pressure_force0))

            if np.abs(np.abs(self._suction_contact_pressure_force[1, 0]) - np.abs(contact_pressure_force0)) / np.abs(contact_pressure_force0)\
                    >= epsilon:

                # Close to wall!
                # Calculate forces for wall close regime calculation

                forces = (self._suction_lower_pressure_force[1, 0]
                          + self._suction_contact_pressure_force[1, 0]
                          - self._suction_gravity_force[1, 0]
                          - self._suction_spring_force[1, 0]
                          - self._suction_damping_force[1, 0]
                          - self._suction_upper_pressure_force[1, 0])

                # Result is the current displacement of the valve

                result = (forces * (self._delta_t**2)
                          / (self._suction_valve_mass + self.suction_spring_mass * (1 / 3.0))
                          + 2.0 * displacement - self._suction_valve_displacement[2, 0])

                # Check: Is the displacement > 0.0?

                if result <= 0.0:

                    # Valve is closed!

                    result = 0.0
                    self.mass_flow_suction_valve[0, 0] = 0.0

                elif result >= self.suction_max_displacement:

                    result = self.suction_max_displacement
                    self.mass_flow_suction_valve[0, 0] = 0.0

                    # Valve is still open
                else:

                    result = result

                # Calculating results
                self._suction_valve_displacement[0, 0] = result
                self._suction_valve_velocity[0, 0] = (result - displacement) / self._delta_t
                self._suction_valve_acceleration[0, 0] = ((result - 2.0 * displacement + self._suction_valve_displacement[2, 0])
                                                          / (self._delta_t**2))
                self.mass_flow_suction_valve[0, 0] = 0.0

                # For close Wall opening the valve has no flow through it

                self._cases.append('Close Wall regime')

                no_flow = True

                return no_flow

            # Check: Close to wall regime?

            else:

                # No! --> Flow regime!
                self.flow_counter = True
                volume_flow = self.calculate_suction_gap_flow(lower_pressure, density)

                reynolds_number = self.calculate_suction_gap_reynolds_number(displacement,
                                                                             volume_flow,
                                                                             viscosity)

                flow_coefficient = self.calculate_suction_dimensionless_coefficient_of_force(reynolds_number,
                                                                                             displacement)

                self._suction_valve_zeta[1, 0] = self.calculate_suction_zeta_value(displacement, reynolds_number)

                self._suction_flow_force[1, 0] = self.calculate_suction_flow_force(flow_coefficient,
                                                                                   density,
                                                                                   lower_pressure,
                                                                                   upper_pressure)

                forces = (self._suction_flow_force[1, 0]
                          - self._suction_gravity_force[1, 0]
                          - self._suction_spring_force[1, 0]
                          - self._suction_damping_force[1, 0]
                          )

                # Flow Regime! --> Volumeflow through Valve allowed!

                no_flow = False

                result = (forces * (self._delta_t**2)
                          / (self._suction_valve_mass + self.suction_spring_mass * (1 / 3.0))
                          + 2.0 * displacement - self._suction_valve_displacement[2, 0])

                # Check: Is Valve Closed?

                if result <= 0.0:

                    # Valve Closed!

                    self.flow_counter = False

                    result = 0.0

                # Check: Is Valve at maximum displacement?

                elif result >= self.suction_max_displacement:

                    # Valve is at maximum allowed displacement

                    result = self.suction_max_displacement

                # Valve is somewhere else!

                else:
                    result = result

                self._suction_valve_displacement[0, 0] = result
                self._suction_valve_velocity[0, 0] = (result - displacement) / self._delta_t
                self._suction_valve_acceleration[0, 0] = ((result - 2.0 * displacement + self._suction_valve_displacement[2, 0])
                                                  / (self._delta_t**2))
                self.calculate_suction_flow(density)

                self._cases.append('Flow regime')

                return no_flow





















































    def calculate_discharge_gravity_force(self, density):

        result = (((self.discharge_valve_density - density)
                   / self.discharge_valve_density)
                  * (self.discharge_valve_mass + self.discharge_spring_mass) * self._g)
        return result

    def calculate_discharge_spring_force(self, displacement):
        """
        Method to calculate the reacting force depending on the displacement of the valve

        :param displacement: Displacement of the valve
        :return: Returns the spring force
        """
        result = self.discharge_spring_force0 + self._discharge_spring_stiffness * displacement

        return result

    def calculate_discharge_lower_pressure_force(self, pressure):
        """


        :param pressure:
        :return:
        """

        result = self.discharge_contact_area * pressure

        return result

    def calculate_discharge_upper_pressure_force(self, pressure):

        result = self.discharge_valve_area * pressure

        return result

    def calculate_discharge_contact_pressure(self, lower_pressure, upper_pressure, displacement, velocity, density, viscosity):

        if displacement <= 1e-7:

            pk = (lower_pressure * ((self.discharge_outer_contact_radius / self.discharge_radius - 1)
                                    / (self.discharge_outer_contact_radius / self.discharge_inner_contact_radius - 1))
                  + upper_pressure * ((1 - self.discharge_inner_contact_radius / self.discharge_radius)
                                      / (1 - self.discharge_inner_contact_radius / self.discharge_outer_contact_radius)))

        elif self.flow_counter:
            pk = (lower_pressure * ((self.discharge_outer_contact_radius / self.discharge_radius - 1)
                                    / (self.discharge_outer_contact_radius / self.discharge_inner_contact_radius - 1))
                  + upper_pressure * ((1 - self.discharge_inner_contact_radius / self.discharge_radius)
                                      / (1 - self.discharge_inner_contact_radius / self.discharge_outer_contact_radius)))

        else:
            pk = (lower_pressure * ((self.discharge_outer_contact_radius / self.discharge_radius - 1)
                                    / (self.discharge_outer_contact_radius / self.discharge_inner_contact_radius - 1))
                  + upper_pressure * ((1 - self.discharge_inner_contact_radius / self.discharge_radius)
                                      / (1 - self.discharge_inner_contact_radius / self.discharge_outer_contact_radius))
                  + (6.0 * velocity * viscosity * density
                     * (self.discharge_outer_contact_radius**2 - self.discharge_inner_contact_radius**2)
                     * (self.discharge_radius - self.discharge_inner_contact_radius)
                     * (self.discharge_outer_contact_radius - self.discharge_radius))
                  / ((displacement**3) * ((np.sin(self._discharge_seat_tilt))**2)
                     * self.discharge_radius * (self.discharge_outer_contact_radius - self.discharge_inner_contact_radius))
                  )

        for i in range(pk.shape[0]):
            if pk[i] <= 2300:
                pk[i] = 2300

        f1 = pk * self.discharge_radius * np.pi

        result = integrate.simps(f1, self.discharge_radius)

        return result

    def calculate_discharge_dampening_force(self, density, velocity, viscosity):

        teta = density * np.abs(velocity) * self._discharge_flow_constant_1\
               + self._discharge_flow_constant_2 * viscosity * density

        result = teta * velocity

        return result

    def calculate_discharge_gap_reynolds_number(self, displacement, volume_flow, viscosity):

        averaged_diameter = (self.discharge_outer_diameter + self.discharge_inner_contact_diameter) / 2.0
        averaged_gap_diameter = averaged_diameter - (displacement / 2.0) * np.sin(2.0 * self._discharge_seat_tilt)
        self._discharge_gap_area[1, 0] = averaged_gap_diameter * np.pi * displacement * np.sin(self._discharge_seat_tilt)
        gap_reynolds_number = 2.0 * np.abs(volume_flow - (averaged_diameter**2)/4.0
                                           * np.pi * self._discharge_valve_velocity[1, 0]) / (viscosity
                                                                                            * np.pi
                                                                                            * (self._discharge_gap_area[1, 0]))

        return gap_reynolds_number

    def calculate_discharge_dimensionless_coefficient_of_force(self, reynolds_number, displacement):
        if reynolds_number == 0.0:
            result = 0.5
        else:
            result = (self._discharge_factor_k0
                      - self._discharge_factor_k1 * (displacement / self.discharge_outer_diameter)
                      - self._discharge_factor_k2 * np.log(reynolds_number))

        return result

    def calculate_discharge_flow_force(self, dimensionless_coefficient, density, lower_pressure, upper_pressure):

        result = dimensionless_coefficient * (lower_pressure - upper_pressure) * (self.discharge_outer_diameter**2) * np.pi / 4.0
        return result

    def calculate_discharge_zeta_value(self, displacement, reynolds_number):

        if reynolds_number > 0.0:
            result = ((self._discharge_friction_factor_a / reynolds_number + self._discharge_friction_factor_c)
                      * (self._discharge_friction_factor_b
                         * np.power(np.abs(self.discharge_inner_contact_diameter / displacement), self._discharge_friction_factor_d) + 1))
        else:
            result = self._discharge_valve_zeta[2, 0]
        return result

    def calculate_discharge_flow(self, no_flow):

        pressure_b = self._pressure[1, -1]
        velocity_b = self._velocity[1, -1]
        friction_b = self._friction[1, -1]
        area_b = self._area[1]
        sos_b = self._sos[1, -1]
        density_b = self.fluid.density(pressure=pressure_b, temperature=None)
        if no_flow:
            self._discharge_volume_flow[0, 0] = 0.0 + self.discharge_valve_area * self._discharge_valve_velocity[0, 0]
        else:

            self._discharge_volume_flow[0, 0] = (self.mass_flow_discharge_valve[0, 0] / density_b
                                                 + self.discharge_valve_area * self._discharge_valve_velocity[0, 0])

        velocity = self._discharge_volume_flow[0, 0] / area_b

        self._pressure[0, -2] = (- density_b * sos_b * velocity_b
                                 + density_b * sos_b * velocity
                                 + pressure_b
                                 + friction_b * self._delta_t * density_b * sos_b)

        return None

    def calculate_discharge_valve(self, lower_pressure, upper_pressure, viscosity, density):

        displacement = self._discharge_valve_displacement[1, 0]
        velocity = self._discharge_valve_velocity[1, 0]
        epsilon = 1e-6
        volume_flow = self.mass_flow_discharge_valve[1, 0]

        # Check: Is valve closed?

        if displacement <= 0:

            # Valve is closed!
            self.flow_counter = False
            # Calculate forces of the closed regime!

            self._discharge_spring_force[1, 0] = self.calculate_discharge_spring_force(displacement)
            self._discharge_gravity_force[1, 0] = self.calculate_discharge_gravity_force(density)
            self._discharge_upper_pressure_force[1, 0] = self.calculate_discharge_upper_pressure_force(upper_pressure)
            self._discharge_lower_pressure_force[1, 0] = self.calculate_discharge_lower_pressure_force(lower_pressure)
            self._discharge_contact_pressure_force[1, 0] = self.calculate_discharge_contact_pressure(lower_pressure,
                                                                                                     upper_pressure,
                                                                                                     0.0,
                                                                                                     0.0,
                                                                                                     density,
                                                                                                     viscosity)

            # Check: Are upper forces bigger than lower forces?

            if self._discharge_gravity_force[1, 0] + self._discharge_spring_force[1, 0] + self._discharge_upper_pressure_force[1, 0]\
                    <= self._discharge_lower_pressure_force[1, 0] + self._discharge_contact_pressure_force[1, 0]:

                # Lower forces are bigger than upper forces --> Valve moves!

                self._discharge_damping_force[1, 0] = self.calculate_discharge_dampening_force(density, velocity, viscosity)

                # Calculate forces for wall close regime calculation

                forces = (self._discharge_lower_pressure_force[1, 0]
                          + self._discharge_contact_pressure_force[1, 0]
                          - self._discharge_gravity_force[1, 0]
                          - self._discharge_spring_force[1, 0]
                          - self._discharge_damping_force[1, 0]
                          - self._discharge_upper_pressure_force[1, 0])

                # Result is the current displacement of the valve

                result = (forces * (self._delta_t**2)
                          / (self._discharge_valve_mass + self.discharge_spring_mass * (1 / 3.0))
                          + 2.0 * displacement - self._discharge_valve_displacement[2, 0])

                if result >= self.discharge_max_displacement:

                    result = self.discharge_max_displacement
                    self.mass_flow_discharge_valve[0, 0] = 0.0

                else:

                    result = result
                    self.mass_flow_discharge_valve[0, 0] = 0.0

                # Calculating results

                self._discharge_valve_displacement[0, 0] = result
                self._discharge_valve_velocity[0, 0] = (result - displacement) / self._delta_t
                self._discharge_valve_acceleration[0, 0] = ((result - 2.0 * displacement + self._discharge_valve_displacement[2, 0])
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
                self._discharge_valve_displacement[0, 0] = result
                self._discharge_valve_velocity[0, 0] = (result - displacement) / self._delta_t
                self._discharge_valve_acceleration[0, 0] = ((result - 2.0 * displacement + self._discharge_valve_displacement[2, 0])
                                                          / (self._delta_t**2))
                self.mass_flow_discharge_valve[0, 0] = 0.0

                # Valve closed now flow through it
                self._cases.append('Valve Closed and stays closed')

                no_flow = True

                return no_flow

        # Check: Is Valve closed?

        else:

            # Valve is OPEN! --> displacement > 0.0!

            contact_pressure_force0 = self.calculate_discharge_contact_pressure(lower_pressure,
                                                                                upper_pressure,
                                                                                displacement,
                                                                                0.0,
                                                                                density,
                                                                                viscosity)

            self._discharge_contact_pressure_force[1, 0] = self.calculate_discharge_contact_pressure(lower_pressure,
                                                                                                     upper_pressure,
                                                                                                     displacement,
                                                                                                     velocity,
                                                                                                     density,
                                                                                                     viscosity)

            self._discharge_spring_force[1, 0] = self.calculate_discharge_spring_force(displacement)
            self._discharge_gravity_force[1, 0] = self.calculate_discharge_gravity_force(density)
            self._discharge_upper_pressure_force[1, 0] = self.calculate_discharge_upper_pressure_force(upper_pressure)
            self._discharge_lower_pressure_force[1, 0] = self.calculate_discharge_lower_pressure_force(lower_pressure)
            self._discharge_damping_force[1, 0] = self.calculate_discharge_dampening_force(density, velocity, viscosity)

            # Check: Close to wall regime?
            self.epsilon.append(np.abs(np.abs(self._discharge_contact_pressure_force[1, 0]) - np.abs(contact_pressure_force0)) / np.abs(contact_pressure_force0))

            if np.abs(np.abs(self._discharge_contact_pressure_force[1, 0]) - np.abs(contact_pressure_force0)) / np.abs(contact_pressure_force0)\
                    >= epsilon:

                # Close to wall!
                # Calculate forces for wall close regime calculation

                forces = (self._discharge_lower_pressure_force[1, 0]
                          + self._discharge_contact_pressure_force[1, 0]
                          - self._discharge_gravity_force[1, 0]
                          - self._discharge_spring_force[1, 0]
                          - self._discharge_damping_force[1, 0]
                          - self._discharge_upper_pressure_force[1, 0])

                # Result is the current displacement of the valve

                result = (forces * (self._delta_t**2)
                          / (self._discharge_valve_mass + self.discharge_spring_mass * (1 / 3.0))
                          + 2.0 * displacement - self._discharge_valve_displacement[2, 0])

                # Check: Is the displacement > 0.0?

                if result <= 0.0:

                    # Valve is closed!

                    result = 0.0
                    self.mass_flow_discharge_valve[0, 0] = 0.0

                elif result >= self.discharge_max_displacement:

                    result = self.discharge_max_displacement
                    self.mass_flow_discharge_valve[0, 0] = 0.0

                    # Valve is still open
                else:

                    result = result

                # Calculating results
                self._discharge_valve_displacement[0, 0] = result
                self._discharge_valve_velocity[0, 0] = (result - displacement) / self._delta_t
                self._discharge_valve_acceleration[0, 0] = ((result - 2.0 * displacement + self._discharge_valve_displacement[2, 0])
                                                            / (self._delta_t**2))
                self.mass_flow_discharge_valve[0, 0] = 0.0

                # For close Wall opening the valve has no flow through it

                self._cases.append('Close Wall regime')

                no_flow = True

                return no_flow

            # Check: Close to wall regime?

            else:

                # No! --> Flow regime!
                self.flow_counter = True
                volume_flow = self.calculate_discharge_gap_flow(lower_pressure, density)

                reynolds_number = self.calculate_discharge_gap_reynolds_number(displacement,
                                                                               volume_flow,
                                                                               viscosity)

                flow_coefficient = self.calculate_discharge_dimensionless_coefficient_of_force(reynolds_number,
                                                                                             displacement)

                self._discharge_valve_zeta[1, 0] = self.calculate_discharge_zeta_value(displacement, reynolds_number)

                self._discharge_flow_force[1, 0] = self.calculate_discharge_flow_force(flow_coefficient,
                                                                                       density,
                                                                                       lower_pressure,
                                                                                       upper_pressure)

                forces = (self._discharge_flow_force[1, 0]
                          - self._discharge_gravity_force[1, 0]
                          - self._discharge_spring_force[1, 0]
                          - self._discharge_damping_force[1, 0]
                          )

                # Flow Regime! --> Volumeflow through Valve allowed!

                no_flow = False

                result = (forces * (self._delta_t**2)
                          / (self._discharge_valve_mass + self.discharge_spring_mass * (1 / 3.0))
                          + 2.0 * displacement - self._discharge_valve_displacement[2, 0])

                # Check: Is Valve Closed?

                if result <= 0.0:

                    # Valve Closed!

                    self.flow_counter = False

                    result = 0.0

                # Check: Is Valve at maximum displacement?

                elif result >= self.discharge_max_displacement:

                    # Valve is at maximum allowed displacement

                    result = self.discharge_max_displacement

                # Valve is somewhere else!

                else:
                    result = result

                self._discharge_valve_displacement[0, 0] = result
                self._discharge_valve_velocity[0, 0] = (result - displacement) / self._delta_t
                self._discharge_valve_acceleration[0, 0] = ((result - 2.0 * displacement + self._discharge_valve_displacement[2, 0])
                                                            / (self._delta_t**2))
                self.calculate_discharge_flow(density)

                self._cases.append('Flow regime')

                return no_flow

















































    def calculate_volume_change(self, t):

        result = self.piston_area * self.calculate_piston_velocity(t)

        return result

    def calculate_volume(self, t):

        result = self.piston_area * self.calculate_piston_stroke(t)

        return result

    def calculate_pump_pressure(self, compressibility, density_pump):

        mass_flow_suction_valve = self.mass_flow_suction_valve[1, 0]

        if self.pump_pressure[1, 0] <= self.fluid.vapor_pressure():
            self.pump_pressure[0, 0] = self.fluid.vapor_pressure()
        else:
            self.pump_pressure[0, 0] = (mass_flow_suction_valve - density_pump *
                                        (self._pump_volume_change[0, 0]
                                         - self.suction_valve_area * self._suction_valve_velocity[1, 0]
                                         + self.discharge_valve_area * self._discharge_valve_velocity[1, 0]))\
                                       / (density_pump / compressibility *
                                          (self.pump_death_volume
                                           + self._pump_volume[1, 0]
                                           - self.suction_valve_area * self._suction_valve_displacement[1, 0]
                                           + self.discharge_valve_area * self._discharge_valve_displacement[1, 0]))

        return None

    def calculate_suction_gap_flow(self, pressure, density):

        gap_velocity = (np.sqrt(np.abs(pressure - self.pump_pressure[0, 0]) * 2.0
                                / (self._suction_valve_zeta[1, 0] * density)) * np.sign(pressure - self.pump_pressure[0, 0]))
        result = gap_velocity * self._suction_gap_area[1, 0]

        return result

    def calculate_next_inner_iteration(self, iteration: int) -> bool:
        """
        Method to do the calculations of the next inner iteration

        :param iteration: Number of the next inner iteration
        :return: Whether this component needs another inner iteration afterwards
        """
        # Get the input fields
        pressure_a = self._pressure[1, 0]
        pressure_b = self._pressure[1, -1]

        compressibility = self.fluid.compressibility()

        # Calculate fluid properties
        density_a = self.fluid.density(pressure=pressure_a, temperature=None)
        density_b = self.fluid.density(pressure=pressure_b, temperature=None)
        viscosity = self.fluid.viscosity(temperature=None, shear_rate=None) / density_a
        density_pump = self.fluid.density(pressure=self.pump_pressure[1, 0])

        # Calculate Volumechange in Pump

        self._pump_volume_change[0, 0] = self.calculate_volume_change(self.time)
        self._pump_volume[0, 0] = self.calculate_volume(self.time)
        self.calculate_pump_pressure(compressibility, density_pump)
        self._suction_gap_flow[0, 0] = self.calculate_suction_gap_flow(pressure_a, density_a)
        self.calculate_suction_valve(pressure_a, self.pump_pressure[0, 0], viscosity, density_a)
        self.calculate_discharge_valve(self.pump_pressure[0, 0], pressure_b, viscosity, density_b)

        return False
