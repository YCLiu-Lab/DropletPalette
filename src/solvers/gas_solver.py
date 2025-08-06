"""
Gas phase solver for droplet combustion simulation

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.
"""

import numpy as np
from src.core import Grid, Surface
import cantera as ct
from .math_utils import ThreePointsScheme, EquationProperty
from .gas_solver_T_evap import GasSolverTEvap
from .gas_solver_Y_evap import GasSolverYEvap



class GasSolver:
    """gas phase solver"""
    
    def __init__(self, grid: Grid, surface: Surface, gas_inf: ct.Solution, flag_fast_solver_for_transient: bool):
        """initialize the gas phase solver
        
        Args:
            grid: grid object
            surface: surface object
            gas_inf: infinite far gas phase state
        """
        self.grid = grid
        self.surface = surface
        self.gas_inf = gas_inf
        self.flag_fast_solver_for_transient = flag_fast_solver_for_transient
        self.three_points_scheme = ThreePointsScheme.create(self.grid.gas_grid.cell_count)
        self.equation_property = EquationProperty.create(self.grid.gas_grid.cell_count, self.grid.gas_array.n_species)


        self.gas_array_iter = ct.SolutionArray(grid.gas_array._phase, shape=grid.gas_array.shape)
        self.gas_array_iter.TPX = grid.gas_array.TPX
        self._relative_velocity = np.zeros(self.grid.gas_grid.cell_count+1)
        self._relative_velocity_right = np.zeros(self.grid.gas_grid.cell_count)
        self._relative_velocity_left = np.zeros(self.grid.gas_grid.cell_count)

        self.three_points_scheme.update_scheme(self.grid.gas_grid.lambda_center, self._relative_velocity) 
        
        # update the equation parameters
        self.equation_property.gas_update_equation_property_T(self.gas_array_iter, self.grid, self.surface, self.gas_inf, self.three_points_scheme)
        self.equation_property.gas_update_equation_property_Y(self.gas_array_iter, self.grid, self.surface, self.gas_inf, self.three_points_scheme, self.gas_array_iter.Y)

        # create the temperature field and mass fraction field solver
        self.T_solver = GasSolverTEvap(grid, surface, gas_inf, self.three_points_scheme, self.equation_property,self.gas_array_iter,self._relative_velocity_right,self._relative_velocity_left,flag_fast_solver_for_transient)
        self.y_solver = GasSolverYEvap(grid, surface, gas_inf, self.three_points_scheme, self.equation_property,self.gas_array_iter,self._relative_velocity_right,self._relative_velocity_left,flag_fast_solver_for_transient)



    def set_velocity(self, velocity: np.ndarray):
        self._relative_velocity[1:] = velocity
        self._relative_velocity_right[:] = self._relative_velocity[1:]
        self._relative_velocity_left[:] = self._relative_velocity[:-1]
        self.update_scheme_and_equation_property()


    def update_scheme_and_equation_property(self):
        self.three_points_scheme.update_scheme(self.grid.gas_grid.lambda_center, self._relative_velocity) 
        # update the equation parameters
        self.equation_property.gas_update_equation_property_T(self.gas_array_iter, self.grid, self.surface, self.gas_inf, self.three_points_scheme)
        self.equation_property.gas_update_equation_property_Y(self.gas_array_iter, self.grid, self.surface, self.gas_inf, self.three_points_scheme, self.gas_array_iter.Y)
