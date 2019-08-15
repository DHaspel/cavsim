import numpy as np


class Pipe:

    def __init__(self, diameter: float, length: float, roughness: float,
                 wall_thickness: float, bulk_modulus: float) -> None:

        self._diameter = diameter
        self._length = length
        self._pressure = np.zeros((3, 10))
        self._velocity = np.zeros((3, 10))
        self._fluid: Fluid()
        self._roughness = roughness
        self._wall_thickness = wall_thickness
        self._bulk_modulus = bulk_modulus
        self._reduced_sound_of_speed = self.calc_reduced_speed_of_sound()
        self._friction = np.zeros((1, 10))

    @property
    def area(self) -> float:
        """
        Function calculates pipe area

        :return: Pipe area
        """
        return (self._diameter**2)/4*np.pi

    @property
    def _reduced_speed_of_sound(self):

        return self._fluid.speedofsound / np.sqrt(1 + (self._fluid.compressibility
                                                       / self._bulk_modulus * self._diameter / self._wall_thickness))

    @property
    def volume(self):

        return self.area*self._length

    def calc_reynolds_number(self, index, time_index=1):

        return self._velocity[time_index, index]*self._diameter/self._fluid.viscosity

    def calc_friction_factor(self, index):

        reynolds_number = self.calc_reynolds_number(index, )

        if reynolds_number == 0:
            friction_factor = 1

        elif 0 < reynolds_number < 2100:
            friction_factor = 64/reynolds_number

        else:
            ff = 10

            err = 0.0001
            f_old = 0

            while err > 1e-12:
                f = 1/np.power(ff, 2)
                ff = -2*np.log10(self._roughness/(3.7*self._diameter))+2.51/(reynolds_number*np.sqrt(f))
                err = np.abs(f-f_old)
                f_old = f

        return friction_factor

    def calc_new_pressure(self, position_index, friction):

        pressure_difference = self._pressure[1, position_index-1]-self._pressure[1, position_index+1]
        velocity_difference = self._velocity[1, position_index-1]+self._velocity[1, position_index+1]
        density_area_factor = self._fluid.density*self.calc_reduced_speed_of_sound()
        self._pressure[0, position_index] = 0.5*((1/density_area_factor)*pressure_difference+velocity_difference -
                                                 self.time_step*(self.friction_difference))

    def calc_new_density(self, position_index):

        return self._fluid.calc_density_constant_model(self._pressure[1, position_index])



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

