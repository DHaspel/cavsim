import numpy as np


class Pipe:

    def __init__(self, diameter: float, length: float, roughness: float,
                 wall_thickness: float, bulk_modulus: float, phi: float) -> None:

        self._diameter = diameter
        self._length = length
        self._pressure = np.zeros((3, 10))
        self._velocity = np.zeros((3, 10))
        self._fluid: Fluid()
        self._roughness = roughness
        self._wall_thickness = wall_thickness
        self._bulk_modulus = bulk_modulus
        self._reduced_speed_of_sound = self.calc_reduced_speed_of_sound()
        self._friction = np.zeros(10)
        self._phi = phi
        self._height = np.sin(self._phi)
        self._timestep = self.solver.timestep
        self._spatialstep = self.solver.spatialstep
        self._gravity = 9.81
        self._density = np.zeros(10)
        self._reynolds_number = np.zeros(1, 10)
        self._friction_steady = np.zeros(1, 10)
        self._friction_unsteady_a = np.zeros(1, 9)
        self._friction_unsteady_b = np.zeros(1, 9)

    @property
    def area(self) -> float:
        """
        Function calculates pipe area

        :return: Pipe area
        """
        return (self._diameter**2)/4*np.pi

    @property
    def _reduced_sound_of_speed(self):
        """
        Function calculates the reduced speed of sound

        :return: reduced speed of sound
        """
        return self._fluid.speedofsound / np.sqrt(1 + (self._fluid.compressibility
                                                       / self._bulk_modulus * self._diameter / self._wall_thickness))

    @property
    def volume(self):
        """
        Property Volume represents the Volume of the Liquid in the whole pipe

        :return: Volume of the inner pipe
        """
        return self.area*self._length

    def calc_reynolds_number(self, time_index=1):
        """
        Function calculates the Reynolds-number of a given space and time index

        :param time_index: Index of the given time of the Simulation
        :return: Reynolds-number
        """
        self._reynolds_number = self._velocity[time_index]*self._diameter/self._fluid.viscosity
        return None

    def calc_friction_factor(self):
        """
        Calculates darcy's friction coefficient of a given spatial location in the pipe

        :param index: Index of the given spatial position in the pipe
        :return: Darcy's friction factor
        """
        for i in range(self._reynolds_number):

            if self._reynolds_number[i] == 0:
                self._friction_factor[i] = 1

            elif 0 < self._reynolds_number[i] < 2100:
                self._friction_factor[i] = 64/self._reynolds_number[i]

            else:
                ff = 10

                err = 0.0001
                f_old = 0

                while err > 1e-12:
                    f = 1/np.power(ff, 2)
                    ff = -2*np.log10(self._roughness/(3.7*self._diameter))+2.51/(reynolds_number*np.sqrt(f))
                    err = np.abs(f-f_old)
                    f_old = f
                    friction_factor[i] = f

        return None

    def calc_current_pressure(self, position_index):
        """
        Calculates the pressure of the current time step

        :param position_index: The spatial index of the pipe
        :return: None
        """
        pressure_sum = self._pressure[1, position_index-1]+self._pressure[1, position_index+1]
        velocity_difference = self._velocity[1, position_index-1]-self._velocity[1, position_index+1]
        friction_difference = self._friction[position_index+1]-self._friction[position_index-1]
        height_difference = self._height[position_index+1]-self._height[position_index-1]
        density_sound_of_speed_factor = self._density[position_index]*self._reduced_speed_of_sound

        self._pressure[0, position_index] = 0.5*((density_sound_of_speed_factor*velocity_difference
                                                  + pressure_sum
                                                  + self._timestep*density_sound_of_speed_factor*friction_difference
                                                  + self._gravity*self._timestep*density_sound_of_speed_factor
                                                  * height_difference))

    def calc_current_velocity(self, position_index):
        """
        Calculates the velocity of the current time step

        :param position_index: The spatial index of the pipe
        :return: None
        """
        pressure_difference = self._pressure[1, position_index-1]-self._pressure[1, position_index+1]
        density_sound_of_speed_factor = self._density[position_index]*self._reduced_speed_of_sound
        velocity_sum = self._velocity[1, position_index-1]+self._velocity[1, position_index+1]
        friction_sum = self._friction[position_index-1]+self._friction[position_index+1]
        height_sum = self._height[position_index-1]+self._height[position_index+1]

        self._velocity[0, position_index] = 0.5*((velocity_sum
                                                  + 1/density_sound_of_speed_factor*pressure_difference
                                                  - friction_sum*self._timestep
                                                  - self._timestep*self._gravity*height_sum))

    def calc_friction_steady(self):
        """
        Calculates the steady friction of the pipe flow of the previous time step
        :return:
        """
        self._friction_steady = (self._friction_factor/(2*self._diameter))\
                                * np.abs(self._velocity[1, :])

    def calc_friction_unsteady_a(self, index):

        if self._reynolds_number[index-1] < 2100:
            factor = 0.00476
        else:
            factor = 7.41 / (np.power(self._reynolds_number[index-1],
                                 np.log10(14.3 / np.power(self._reynolds_number[index-1], 0.05))))

        brunone_factor = np.sqrt(factor)/2
        time_difference = (self._velocity[1, index-1]-self._velocity[2, index-1])/self._timestep
        spatial_difference = (self._velocity[1, index]-self._velocity[1, index-1])/self._spatialstep
        unsteady_friction = brunone_factor*(time_difference
                                            + self._reduced_speed_of_sound
                                            * np.sign(self._velocity[1, index-1]*spatial_difference)
                                            * spatial_difference)
        return unsteady_friction

    def calc_friction_unsteady_b(self, index):

        if self._reynolds_number[index+1] < 2100:
            factor = 0.00476
        else:
            factor = 7.41 / (np.power(self._reynolds_number[index+1],
                                 np.log10(14.3 / np.power(self._reynolds_number[index+1], 0.05))))
        brunone_factor = np.sqrt(factor) / 2
        time_difference = (self._velocity[1, index+1]-self._velocity[2, index+1])/self._timestep
        spatial_difference = (self._velocity[1, index+1]-self._velocity[1, index])/self._spatialstep
        unsteady_friction = brunone_factor*(time_difference
                                            + self._reduced_speed_of_sound
                                            * np.sign(self._velocity[1, index+1]*spatial_difference)
                                            * spatial_difference)
        return unsteady_friction

    def calc_new_density(self, position_index):
        """
        Calculates the density of the current time step

        :param position_index: The spatial index of the pipe
        :return: density of the current time step
        """
        return self._fluid.calc_density_constant_model(self._pressure[1, position_index])

    def prepare_new_time_step(self):

        self._pressure[1:2, :] = self._pressure[0:1, :]
        self._velocity[1:2, :] = self._velocity[0:1, :]
        self._density[:] = self.fluid.calc_density_constant_model(self._pressure[1, :])

class Fluid:


    @property
    def viscosity(self):
        pass

    @property
    def compressibility(self):
        pass

    @property
    def density(self):
        pass

    @staticmethod
    def norm_density():

        return 1e3

    @staticmethod
    def norm_pressure():

        return 1e5

    def calc_density_constant_model(self, pressure):

        return self.norm_density()*np.exp((pressure-self.norm_pressure())/self.compressibility)


pipe_1 = Pipe(0.01, 20.0, 0.0001)

