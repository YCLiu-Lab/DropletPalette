"""
grid system module - manage the grid system of the droplet combustion simulation

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.

main classes:
- GridParameters: grid parameter configuration
- PhaseGrid: phase grid data structure
- Grid: grid system management
"""

import numpy as np
import cantera as ct
from scipy.interpolate import interp1d
from typing import Optional
from dataclasses import dataclass, field
from .runtime import Runtime
from src.solution import Liquid

@dataclass
class GridParameters:
    """grid parameter configuration class
    
    Attributes:
        droplet_radius: current droplet radius [m]
        droplet_radius_init: initial droplet radius [m]
        r_inf: calculation domain radius [m]
        gas_grid_total_ratio: gas phase grid equal ratio sum ratio [-]
        gas_cell_count: gas phase grid count [-]
        liquid_cell_count: liquid phase grid count [-]
        bias: gas phase grid bias factor [-]
    """
    droplet_radius: float = 5e-4         # current droplet radius [m]
    droplet_radius_init: float = 5e-4    # initial droplet radius [m]
    r_inf: float = field(init=False)     # calculation domain radius [m]
    gas_grid_total_ratio: float = field(init=False)  # gas phase grid equal ratio sum ratio [-]
    gas_cell_count: int = 200            # gas phase grid count [-]
    liquid_cell_count: int = 40          # liquid phase grid count [-]
    bias: float = 1.05                   # gas phase grid bias factor [-]
    
    def __post_init__(self):
        """initialize the calculation parameters"""
        boundary_ratio = 200.0  # calculation domain radius/droplet radius [-]
        self.r_inf = boundary_ratio * self.droplet_radius_init
        self.gas_grid_total_ratio = (1 - self.bias ** self.gas_cell_count) / (1 - self.bias)

@dataclass
class PhaseGrid:
    """phase grid data structure
    
    Attributes:
        cell_count: grid count
        positions_right_boundary: right boundary position
        positions_left_boundary: left boundary position
        positions_volume_centers: volume center position
        volumes: control volume volume
        areas_right_boundary: right boundary area
        areas_left_boundary: left boundary area
        lambda_center: center difference lambda value
        old_*: previous time corresponding attribute
    """
    __slots__ = ('cell_count', 'positions_right_boundary', 'positions_left_boundary',
                'positions_volume_centers', 'volumes',
                'areas_right_boundary', 'areas_left_boundary', 'lambda_center',
                'old_positions_right_boundary', 'old_positions_left_boundary',
                'old_positions_volume_centers', 'old_volumes')
    
    cell_count: int
    positions_right_boundary: np.ndarray
    positions_left_boundary: np.ndarray
    positions_volume_centers: np.ndarray
    volumes: np.ndarray
    areas_right_boundary: np.ndarray
    areas_left_boundary: np.ndarray
    lambda_center: np.ndarray
    old_positions_right_boundary: np.ndarray
    old_positions_left_boundary: np.ndarray
    old_positions_volume_centers: np.ndarray
    old_volumes: np.ndarray

class Grid:
    """grid system management class
    
    functions:
    - grid initialization and update
    - grid geometry attribute calculation
    - spatial discretization format update
    - state interpolation and saving between previous and current time
    """
    __slots__ = ('params', 'liquid_array', 'old_liquid_array',
                'gas_array', 'old_gas_array', 'liquid_grid', 'gas_grid', 'runtime',
                'liquid_00', 'liquid_surface', 'gas_surface', 'gas_inf')
    
    def __init__(self, liquid_array, old_liquid_array, gas_array: ct.SolutionArray, 
                 old_gas_array: ct.SolutionArray, params: GridParameters, runtime: Runtime,
                 liquid_00: Optional[Liquid] = None,
                 liquid_surface: Optional[Liquid] = None,
                 gas_surface: Optional[ct.Solution] = None,
                 gas_inf: Optional[ct.Solution] = None):
        """initialize the grid system"""
        self.params = params
        self.liquid_array = liquid_array
        self.old_liquid_array = old_liquid_array
        self.gas_array = gas_array
        self.old_gas_array = old_gas_array
        self.runtime = runtime
        
        # initialize the boundary conditions
        self.liquid_00 = liquid_00
        self.liquid_surface = liquid_surface
        self.gas_surface = gas_surface
        self.gas_inf = gas_inf
        
        # create the gas and liquid phase grids
        self.liquid_grid = self._create_phase_grid(params.liquid_cell_count)
        self.gas_grid = self._create_phase_grid(params.gas_cell_count)
        
        # initialize the grid
        self.initialize_grid()
    
    def _create_phase_grid(self, cell_count: int) -> PhaseGrid:
        """create the phase grid data structure"""
        return PhaseGrid(
            cell_count=cell_count,
            positions_right_boundary=np.zeros(cell_count),
            positions_left_boundary=np.zeros(cell_count),
            positions_volume_centers=np.zeros(cell_count),
            volumes=np.zeros(cell_count),
            areas_right_boundary=np.zeros(cell_count),
            areas_left_boundary=np.zeros(cell_count),
            lambda_center=np.zeros(cell_count),
            old_positions_right_boundary=np.zeros(cell_count),
            old_positions_left_boundary=np.zeros(cell_count),
            old_positions_volume_centers=np.zeros(cell_count),
            old_volumes=np.zeros(cell_count)
        )
    
    def initialize_grid(self):
        """initialize the grid system"""
        # calculate the gas phase grid geometry attributes
        unit_distance = (self.params.r_inf - self.params.droplet_radius) / self.params.gas_grid_total_ratio
        gas_drs = unit_distance * np.power(self.params.bias, np.arange(self.gas_grid.cell_count))
        self.gas_grid.positions_right_boundary = np.cumsum(gas_drs) + self.params.droplet_radius
        self.gas_grid.positions_left_boundary = np.concatenate([[self.params.droplet_radius], self.gas_grid.positions_right_boundary[:-1]])
        self._calculate_grid_geometry(self.gas_grid)

        # calculate the liquid phase grid geometry attributes
        dr = self.params.droplet_radius / self.liquid_grid.cell_count
        self.liquid_grid.positions_right_boundary = np.arange(1, self.liquid_grid.cell_count + 1) * dr
        self.liquid_grid.positions_left_boundary = np.concatenate([[0.0], self.liquid_grid.positions_right_boundary[:-1]])
        self._calculate_grid_geometry(self.liquid_grid)

        # update the spatial discretization format
        self._update_numerical_discretization()
    
    def reset_grid(self):
        """reset the grid system"""
        self.save_current_state()
        self._update_grid_geometry_gas()
        self._update_grid_geometry_liquid()
        self._update_numerical_discretization()
        self._update_gas_liquid()
    
    def _update_grid_geometry_gas(self):
        """update the gas phase grid geometry attributes"""
        unit_distance = (self.params.r_inf - self.params.droplet_radius) / self.params.gas_grid_total_ratio
        gas_drs = unit_distance * np.power(self.params.bias, np.arange(self.gas_grid.cell_count))
        self.gas_grid.positions_right_boundary = np.cumsum(gas_drs) + self.params.droplet_radius
        self.gas_grid.positions_left_boundary = np.concatenate([[self.params.droplet_radius], self.gas_grid.positions_right_boundary[:-1]])
        self._calculate_grid_geometry(self.gas_grid)
    
    def _update_grid_geometry_liquid(self):
        """update the liquid phase grid geometry attributes"""
        dr = self.params.droplet_radius / self.liquid_grid.cell_count
        self.liquid_grid.positions_right_boundary = np.arange(1, self.liquid_grid.cell_count + 1) * dr
        self.liquid_grid.positions_left_boundary = np.concatenate([[0.0], self.liquid_grid.positions_right_boundary[:-1]])
        self._calculate_grid_geometry(self.liquid_grid)
    
    def _calculate_grid_geometry(self, phase: PhaseGrid):
        """calculate the grid geometry attributes"""
        phase.areas_right_boundary = phase.positions_right_boundary ** 2
        phase.areas_left_boundary = phase.positions_left_boundary ** 2
        phase.volumes = (phase.positions_right_boundary**3 - phase.positions_left_boundary**3) / 3
        phase.positions_volume_centers = 3 * (phase.positions_right_boundary**4 - phase.positions_left_boundary**4) / \
                          (4 * (phase.positions_right_boundary**3 - phase.positions_left_boundary**3))
    
    def _update_numerical_discretization(self):
        """update the spatial discretization format"""
        # calculate the liquid phase spatial discretization format
        self.liquid_grid.lambda_center = np.zeros_like(self.liquid_grid.positions_volume_centers)
        self.liquid_grid.lambda_center[:-1] = (self.liquid_grid.positions_right_boundary[:-1] - self.liquid_grid.positions_volume_centers[:-1]) / \
            (self.liquid_grid.positions_volume_centers[1:] - self.liquid_grid.positions_volume_centers[:-1])
        
        # calculate the gas phase spatial discretization format
        self.gas_grid.lambda_center = np.zeros_like(self.gas_grid.positions_volume_centers)
        self.gas_grid.lambda_center[:-1] = (self.gas_grid.positions_right_boundary[:-1] - self.gas_grid.positions_volume_centers[:-1]) / \
            (self.gas_grid.positions_volume_centers[1:] - self.gas_grid.positions_volume_centers[:-1])
    
    def _update_gas_liquid(self):
        """
        update the gas and liquid state arrays
        note that the update method of the gas phase is to directly use the volume interpolation, while the update method of the liquid phase is to use the volume ratio interpolation.
        """
        # calculate the cumulative volume
        old_cumulative_volumes_gas = np.cumsum(self.gas_grid.old_volumes)
        new_cumulative_volumes_gas = np.cumsum(self.gas_grid.volumes)
        old_cumulative_volumes_liquid = np.cumsum(self.liquid_grid.old_volumes)
        new_cumulative_volumes_liquid = np.cumsum(self.liquid_grid.volumes)
        
        # calculate the total volume
        old_total_volume_liquid = old_cumulative_volumes_liquid[-1]
        new_total_volume_liquid = new_cumulative_volumes_liquid[-1]
        
        # calculate the relative cumulative volume position
        old_normalized_positions_liquid = old_cumulative_volumes_liquid / old_total_volume_liquid
        new_normalized_positions_liquid = new_cumulative_volumes_liquid / new_total_volume_liquid
        
        # update the gas phase state

        # add the boundary condition on the left
        old_cumulative_volumes_gas_extended = np.concatenate([[0.0], old_cumulative_volumes_gas])
        old_gas_T_extended = np.concatenate([[self.gas_surface.T], self.old_gas_array.T])
        
        T_interp = interp1d(old_cumulative_volumes_gas_extended, old_gas_T_extended,
                           kind='linear', bounds_error=False,
                           fill_value=(self.gas_surface.T, self.gas_inf.T))
        T_new = T_interp(new_cumulative_volumes_gas)
        
        # add the boundary condition on the left
        old_gas_Y_extended = np.vstack([self.gas_surface.Y, self.old_gas_array.Y])
        
        Y_interp = interp1d(
            old_cumulative_volumes_gas_extended, old_gas_Y_extended,
            kind='linear', bounds_error=False,
            fill_value=(self.gas_surface.Y, self.gas_inf.Y),
            axis=0
        )
        Y_new = Y_interp(new_cumulative_volumes_gas)
        self.gas_array.TPY = (T_new, self.old_gas_array.P, Y_new)
        
        # update the liquid phase state
        # add the boundary condition on the left
        old_normalized_positions_liquid_extended = np.concatenate([[0.0], old_normalized_positions_liquid])
        old_liquid_T_extended = np.concatenate([[self.liquid_00.temperature], self.old_liquid_array.T])
        
        T_interp = interp1d(old_normalized_positions_liquid_extended, old_liquid_T_extended,
                           kind='linear', bounds_error=False,
                           fill_value=(self.liquid_00.temperature, self.liquid_surface.temperature))
        T_new = T_interp(new_normalized_positions_liquid)
        
        # add the boundary condition on the left
        old_liquid_Y_extended = np.vstack([self.liquid_00.mass_fraction, self.old_liquid_array.Y])
        
        Y_interp = interp1d(
            old_normalized_positions_liquid_extended, old_liquid_Y_extended,
            kind='linear', bounds_error=False,
            fill_value=(self.liquid_00.mass_fraction, self.liquid_surface.mass_fraction),
            axis=0
        )
        Y_new = Y_interp(new_normalized_positions_liquid)
        self.liquid_array.TPY(T_new, self.old_liquid_array.P, Y_new)
    
    def save_current_state(self):
        """save the current grid state"""
        # save the gas and liquid phase grid state
        for phase in [self.liquid_grid, self.gas_grid]:
            np.copyto(phase.old_positions_right_boundary, phase.positions_right_boundary)
            np.copyto(phase.old_positions_left_boundary, phase.positions_left_boundary)
            np.copyto(phase.old_positions_volume_centers, phase.positions_volume_centers)
            np.copyto(phase.old_volumes, phase.volumes)
        
        # save the gas and liquid state arrays
        self.old_liquid_array.TPY(
            temperature=self.liquid_array.T,
            pressure=self.liquid_array.P,
            mass_fraction=self.liquid_array.Y
        )
        self.old_gas_array.TPX = self.gas_array.TPX