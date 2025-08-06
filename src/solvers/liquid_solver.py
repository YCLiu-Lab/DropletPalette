"""
liquid solver module

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.
"""

import numpy as np
import cantera as ct
from src.core import Grid, Surface
from src.solution import Liquid, LiquidArray
from .math_utils import ThreePointsScheme, EquationProperty
from .liquid_solver_T_evap import LiquidSolverTEvap
from .liquid_solver_Y_evap import LiquidSolverYEvap


class LiquidSolver:
    """liquid phase solver"""
    
    def __init__(self, grid: Grid, liquid_00: Liquid, surface: Surface, solver_type: str = 'ITCID'):
        """initialize the liquid phase solver
        
        Args:
            grid: grid object
            liquid_00: liquid phase state at the center of the droplet (left boundary)
            surface: surface object (right boundary)
            solver_type: solver type, optional values are 'ITCID' (infinite heat and mass transfer), 'ITCFD' (infinite heat and finite mass transfer), or 'FTCFD' (finite heat and mass transfer)
        """
        self.grid = grid
        self.liquid_00 = liquid_00
        self.surface = surface
        self.solver_type = solver_type

        # create the three points scheme and equation property object
        self.three_points_scheme = ThreePointsScheme.create(self.grid.liquid_grid.cell_count)
        self.equation_property = EquationProperty.create(self.grid.liquid_grid.cell_count, len(self.grid.liquid_array.liquids[0].composition))

        # create the iteration array
        self.liquid_array_iter = LiquidArray(self.grid.liquid_array.liquids[0], self.grid.liquid_grid.cell_count)
        self._relative_velocity = np.zeros(self.grid.liquid_grid.cell_count+1)
        # update the three points scheme parameters
        self.three_points_scheme.update_scheme(
            self.grid.liquid_grid.lambda_center, 
            self._relative_velocity
        ) 
        
        # update the equation property
        self.equation_property.liquid_update_equation_property(
            self.liquid_array_iter, 
            self.grid, 
            self.liquid_00,
            self.surface, 
            self.three_points_scheme, 
            self.liquid_array_iter.Y
        )

        # create the temperature field and mass fraction field solver
        self.T_solver = LiquidSolverTEvap(
            grid=self.grid,
            surface=self.surface,
            liquid_00=self.liquid_00,
            three_points_scheme=self.three_points_scheme,
            equation_property=self.equation_property,
            liquid_array_iter=self.liquid_array_iter
        )
        
        self.y_solver = LiquidSolverYEvap(
            grid=self.grid,
            surface=self.surface,
            liquid_00=self.liquid_00,
            three_points_scheme=self.three_points_scheme,
            equation_property=self.equation_property,
            liquid_array_iter=self.liquid_array_iter,
        )

    def update_scheme_parameters(self):
        """update the numerical scheme parameters"""
        # update the three points scheme parameters
        self.three_points_scheme.update_scheme(
            self.grid.liquid_grid.lambda_center, 
            self._relative_velocity
        ) 
        
        # update the equation property
        self.equation_property.liquid_update_equation_property(
            self.liquid_array_iter_last, 
            self.grid, 
            self.liquid_00,
            self.surface, 
            self.three_points_scheme, 
            self.liquid_array_iter_last.Y
        )