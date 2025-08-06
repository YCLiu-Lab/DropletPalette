"""
math utils module

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.
"""

from dataclasses import dataclass
import numpy as np
import cantera as ct
from numba import jit
import warnings
from src.core import Grid, Surface
from src.solution import Liquid, LiquidArray

# three points scheme class, manage various numerical discretization methods, when the velocity changes, the five points scheme coefficients need to be recalculated
# equation property class, calculate the parameters that appear during the numerical calculation, when the gas and liquid phase states change, the equation property class needs to be recalculated, including the parameters for calculating the energy equation and the mass conservation equation
@dataclass(eq=False, order=False, unsafe_hash=False)
class ThreePointsScheme:
    """three points scheme class, manage various numerical discretization methods
    
    Attributes:
        cell_count (int): number of grid cells
        scheme_center_left (np.ndarray): left boundary central difference coefficient [cell_count, 3]
        scheme_center_right (np.ndarray): right boundary central difference coefficient [cell_count, 3]
        scheme_upwind_left (np.ndarray): left boundary first order upwind coefficient [cell_count, 3]
        scheme_upwind_right (np.ndarray): right boundary first order upwind coefficient [cell_count, 3]
    """
    __slots__ = ('cell_count', 'scheme_center_left', 'scheme_center_right',
                'scheme_upwind_left', 'scheme_upwind_right')
    
    cell_count: int
    scheme_center_left: np.ndarray
    scheme_center_right: np.ndarray
    scheme_upwind_left: np.ndarray
    scheme_upwind_right: np.ndarray

    
    @classmethod
    def create(cls, cell_count: int) -> 'ThreePointsScheme':
        """create the three points scheme data structure"""
        return cls(
            cell_count=cell_count,
            scheme_center_left=np.zeros((cell_count, 3)),
            scheme_center_right=np.zeros((cell_count, 3)),
            scheme_upwind_left=np.zeros((cell_count, 3)),
            scheme_upwind_right=np.zeros((cell_count, 3))
        )
    
    def update_scheme(self, lambda_center: np.ndarray, relative_velocity_all: np.ndarray):
        """update the numerical scheme coefficients
        
        Args:
            lambda_center: central difference coefficient
            relative_velocity_all: relative velocity (including the velocity at the left boundary)
        """
        # initialize the array
        self.scheme_center_left.fill(0)
        self.scheme_center_right.fill(0)
        self.scheme_upwind_left.fill(0)
        self.scheme_upwind_right.fill(0)
        
        # left boundary:
        self.scheme_center_left[0, :] = [1/2, 1/2, 0]  # the first grid uses the first order format
        self.scheme_center_left[1:, 0] = 1 - lambda_center[:-1]  # the middle term
        self.scheme_center_left[1:, 1] = lambda_center[:-1]  # the right term
        
        # right boundary:
        # - the last grid: [0, 1, 0] (first order format)
        # - the other grids should be [0, 1-λ, λ]
        self.scheme_center_right[:-1, 1] = 1 - lambda_center[:-1] # the left term
        self.scheme_center_right[:-1, 2] = lambda_center[:-1]  # the middle term
        self.scheme_center_right[-1, :] = [0, 1/2, 1/2]  # the last grid uses the first order format
        
        # update the first order upwind scheme coefficients
        # left boundary upwind scheme:
        # - velocity >= 0: [1, 0, 0] (use the left point)
        # - velocity < 0: [0, 1, 0] (use the current point)
        for i in range(self.cell_count):
            if relative_velocity_all[i] >= 0:
                self.scheme_upwind_left[i, 0] = 1  # use the left point
            else:
                self.scheme_upwind_left[i, 1] = 1  # use the current point
        
        # right boundary upwind scheme:
        # - velocity >= 0: [0, 1, 0] (use the current point)
        # - velocity < 0: [0, 0, 1] (use the right point)
        for i in range(self.cell_count):
            if relative_velocity_all[i+1] >= 0:
                self.scheme_upwind_right[i, 1] = 1  # use the current point
            else:
                self.scheme_upwind_right[i, 2] = 1  # use the right point


@dataclass(eq=False, order=False, unsafe_hash=False)
class EquationProperty:
    """parameter calculation class, calculate the parameters that appear during the numerical calculation"""
    __slots__ = ('specific_enthalpy_left_upwind', 'specific_enthalpy_right_upwind',          
                 'density_left_upwind', 'density_right_upwind',
                 'thermal_conductivity_left_center', 'thermal_conductivity_right_center',
                 'rho_diffusivity_left_center', 'rho_diffusivity_right_center',
                 'temperature_left_upwind', 'temperature_right_upwind',
                 'mass_fraction_left_upwind', 'mass_fraction_right_upwind',
                 'temperature_gradient_left', 'temperature_gradient_right',
                 'mass_fraction_gradient_left', 'mass_fraction_gradient_right',
                 'mass_flux_diffusion', 'mass_flux_convection',
                 'energy_flux_diffusion', 'energy_flux_convection')
    
    specific_enthalpy_left_upwind: np.ndarray
    specific_enthalpy_right_upwind: np.ndarray
    density_left_upwind: np.ndarray
    density_right_upwind: np.ndarray

    thermal_conductivity_left_center: np.ndarray
    thermal_conductivity_right_center: np.ndarray
    rho_diffusivity_left_center: np.ndarray
    rho_diffusivity_right_center: np.ndarray
    
    temperature_left_upwind: np.ndarray
    temperature_right_upwind: np.ndarray
    mass_fraction_left_upwind: np.ndarray
    mass_fraction_right_upwind: np.ndarray

    temperature_gradient_left: np.ndarray
    temperature_gradient_right: np.ndarray
    mass_fraction_gradient_left: np.ndarray
    mass_fraction_gradient_right: np.ndarray
    
    mass_flux_diffusion: np.ndarray
    mass_flux_convection: np.ndarray
    energy_flux_diffusion: np.ndarray
    energy_flux_convection: np.ndarray

    @classmethod
    def create(cls, cell_count: int, n_species: int) -> 'EquationProperty':
        return cls(
            specific_enthalpy_left_upwind=np.zeros(cell_count),
            specific_enthalpy_right_upwind=np.zeros(cell_count),
            density_left_upwind=np.zeros(cell_count),
            density_right_upwind=np.zeros(cell_count),
            thermal_conductivity_left_center=np.zeros(cell_count),
            thermal_conductivity_right_center=np.zeros(cell_count),
            rho_diffusivity_left_center=np.zeros((cell_count,n_species)),
            rho_diffusivity_right_center=np.zeros((cell_count,n_species)),
            temperature_left_upwind=np.zeros(cell_count),
            temperature_right_upwind=np.zeros(cell_count),
            mass_fraction_left_upwind=np.zeros((cell_count,n_species)),
            mass_fraction_right_upwind=np.zeros((cell_count,n_species)),
            temperature_gradient_left=np.zeros(cell_count),
            temperature_gradient_right=np.zeros(cell_count),
            mass_fraction_gradient_left=np.zeros((cell_count,n_species)),
            mass_fraction_gradient_right=np.zeros((cell_count,n_species)),
            mass_flux_diffusion = np.zeros((cell_count,n_species)),
            mass_flux_convection = np.zeros((cell_count,n_species)),
            energy_flux_diffusion = np.zeros(cell_count),
            energy_flux_convection = np.zeros(cell_count)
        )
    


    def check_all_arrays(self):
        """check all arrays for negative values and emit warnings"""
        arrays_to_check = {
            "specific_enthalpy_left_upwind": self.specific_enthalpy_left_upwind,
            "specific_enthalpy_right_upwind": self.specific_enthalpy_right_upwind,
            "density_left_upwind": self.density_left_upwind,
            "density_right_upwind": self.density_right_upwind,
            "thermal_conductivity_left_center": self.thermal_conductivity_left_center,
            "thermal_conductivity_right_center": self.thermal_conductivity_right_center,
            "rho_diffusivity_left_center": self.rho_diffusivity_left_center,
            "rho_diffusivity_right_center": self.rho_diffusivity_right_center,
            "temperature_left_upwind": self.temperature_left_upwind,
            "temperature_right_upwind": self.temperature_right_upwind,
            "mass_fraction_left_upwind": self.mass_fraction_left_upwind,
            "mass_fraction_right_upwind": self.mass_fraction_right_upwind
        }
        
        for name, array in arrays_to_check.items():
            if np.any(array < 0):
                # find all negative values
                negative_indices = np.where(array < 0)
                if len(array.shape) == 1:
                    indices_str = f"location: {negative_indices[0]}"
                else:
                    indices_str = f"location: {list(zip(*negative_indices))}"
                warnings.warn(f"negative values detected in {name}, minimum value: {np.min(array)}, {indices_str}")

    def check_rho_diffusivity_zeros(self):
        """check for 0 values in the diffusion coefficients"""
        if np.any(self.rho_diffusivity_left_center == 0) or np.any(self.rho_diffusivity_right_center == 0):
            warnings.warn("0 values detected in the diffusion coefficients")

    def gas_update_equation_property_T(self, gas_array_iter: ct.SolutionArray, grid: Grid, surface: 'Surface', gas_inf: ct.Solution, three_points_scheme: ThreePointsScheme):
        """update the parameters related to temperature in the gas phase equation
        
        Args:
            gas_array_iter: gas phase state array
            grid: grid object
            surface: surface object
            gas_inf: infinite far gas phase state
            three_points_scheme: three points scheme coefficients
        """
        # update the specific enthalpy
        self.specific_enthalpy_left_upwind = calculate_three_point_array(
            gas_array_iter.cp_mass * gas_array_iter.density_mass,
            surface.gas_surface.cp_mass * surface.gas_surface.density_mass,
            gas_inf.cp_mass * gas_inf.density_mass,
            three_points_scheme.scheme_upwind_left,
            grid.gas_grid.cell_count
        )
        self.specific_enthalpy_right_upwind = calculate_three_point_array(
            gas_array_iter.cp_mass * gas_array_iter.density_mass,
            surface.gas_surface.cp_mass * surface.gas_surface.density_mass,
            gas_inf.cp_mass * gas_inf.density_mass,
            three_points_scheme.scheme_upwind_right,
            grid.gas_grid.cell_count
        )

        # update the thermal conductivity
        self.thermal_conductivity_left_center = calculate_three_point_array(
            gas_array_iter.thermal_conductivity,
            surface.gas_surface.thermal_conductivity,
            gas_inf.thermal_conductivity,
            three_points_scheme.scheme_center_left,
            grid.gas_grid.cell_count
        )
        self.thermal_conductivity_right_center = calculate_three_point_array(
            gas_array_iter.thermal_conductivity,
            surface.gas_surface.thermal_conductivity,
            gas_inf.thermal_conductivity,
            three_points_scheme.scheme_center_right,
            grid.gas_grid.cell_count
        )

        # update the temperature
        self.temperature_left_upwind = calculate_three_point_array(
            gas_array_iter.T,
            surface.gas_surface.T,
            gas_inf.T,
            three_points_scheme.scheme_upwind_left,
            grid.gas_grid.cell_count
        )
        self.temperature_right_upwind = calculate_three_point_array(
            gas_array_iter.T,
            surface.gas_surface.T,
            gas_inf.T,
            three_points_scheme.scheme_upwind_right,
            grid.gas_grid.cell_count
        )

        # calculate the temperature gradient
        self.temperature_gradient_left, self.temperature_gradient_right = calculate_temperature_gradients(
            gas_array_iter.T, grid.gas_grid.positions_volume_centers,
            surface.gas_surface.T, gas_inf.T,
            grid.params.droplet_radius, grid.params.r_inf
        )

        # check the negative values in the temperature related arrays
        for name, array in {
            "specific_enthalpy_left_upwind": self.specific_enthalpy_left_upwind,
            "specific_enthalpy_right_upwind": self.specific_enthalpy_right_upwind,
            "thermal_conductivity_left_center": self.thermal_conductivity_left_center,
            "thermal_conductivity_right_center": self.thermal_conductivity_right_center,
            "temperature_left_upwind": self.temperature_left_upwind,
            "temperature_right_upwind": self.temperature_right_upwind
        }.items():
            if np.any(array < 0):
                negative_indices = np.where(array < 0)
                indices_str = f"location: {negative_indices[0]}"
                warnings.warn(f"negative values detected in {name}, minimum value: {np.min(array)}, {indices_str}")
        
   
    def gas_update_equation_property_Y_fast(self, gas_array_iter: ct.SolutionArray, grid: Grid, surface: 'Surface', gas_inf: ct.Solution, three_points_scheme: ThreePointsScheme, mass_fraction: np.ndarray):
        """update the parameters related to mass fraction in the gas phase equation
        
        Args:
            gas_array_iter: gas phase state array
            grid: grid object
            surface: surface object
        """
        # update the mass fraction
        self.mass_fraction_left_upwind = calculate_multispecies_three_point_array(
            mass_fraction,
            surface.gas_surface.Y,
            gas_inf.Y,
            three_points_scheme.scheme_upwind_left,
            grid.gas_grid.cell_count,
            gas_array_iter.n_species
        )
        self.mass_fraction_right_upwind = calculate_multispecies_three_point_array(
            mass_fraction,
            surface.gas_surface.Y,
            gas_inf.Y,
            three_points_scheme.scheme_upwind_right,
            grid.gas_grid.cell_count,
            gas_array_iter.n_species
        )

        # calculate the mass fraction gradient
        self.mass_fraction_gradient_left, self.mass_fraction_gradient_right = calculate_species_gradients(
            mass_fraction, grid.gas_grid.positions_volume_centers,
            surface.gas_surface.Y, gas_inf.Y,
            grid.params.droplet_radius, grid.params.r_inf
        )
        # check the negative values in the mass fraction related arrays
        for name, array in {
            "mass_fraction_left_upwind": self.mass_fraction_left_upwind,
            "mass_fraction_right_upwind": self.mass_fraction_right_upwind
        }.items():
            if np.any(array < -1e-3):
                negative_indices = np.where(array < -1e-3)
                if len(array.shape) == 1:
                    indices_str = f"location: {negative_indices[0]}"
                else:
                    indices_str = f"location: {list(zip(*negative_indices))}"
                warnings.warn(f"negative values detected in {name}, minimum value: {np.min(array)}, {indices_str}")

    def gas_update_equation_property_Y(self, gas_array_iter: ct.SolutionArray, grid: Grid, surface: 'Surface', gas_inf: ct.Solution, three_points_scheme: ThreePointsScheme, mass_fraction: np.ndarray):
        """update the parameters related to mass fraction in the gas phase equation
        
        Args:
            gas_array_iter: gas phase state array
            grid: grid object
            surface: surface object
            gas_inf: infinite far gas phase state
            three_points_scheme: three points scheme coefficients
            mass_fraction: mass fraction array
        """
        # calculate the product of density and diffusion coefficient
        rho_diffusivity = gas_array_iter.density_mass[:, np.newaxis] * gas_array_iter.mix_diff_coeffs_mass
        rho_diffusivity_inf = gas_inf.density_mass * gas_inf.mix_diff_coeffs_mass
        rho_diffusivity_surface = surface.gas_surface.density_mass * surface.gas_surface.mix_diff_coeffs_mass
        # update the density
        self.density_left_upwind = calculate_three_point_array(
            gas_array_iter.density_mass,
            surface.gas_surface.density_mass,
            gas_inf.density_mass,
            three_points_scheme.scheme_upwind_left,
            grid.gas_grid.cell_count
        )
        self.density_right_upwind = calculate_three_point_array(
            gas_array_iter.density_mass,
            surface.gas_surface.density_mass,
            gas_inf.density_mass,
            three_points_scheme.scheme_upwind_right,
            grid.gas_grid.cell_count
        )

        # update the diffusion coefficient
        self.rho_diffusivity_left_center = calculate_multispecies_three_point_array(
            rho_diffusivity,
            rho_diffusivity_surface,
            rho_diffusivity_inf,
            three_points_scheme.scheme_center_left,
            grid.gas_grid.cell_count,
            gas_array_iter.n_species
        )
        self.rho_diffusivity_right_center = calculate_multispecies_three_point_array(
            rho_diffusivity,
            rho_diffusivity_surface,
            rho_diffusivity_inf,
            three_points_scheme.scheme_center_right,
            grid.gas_grid.cell_count,
            gas_array_iter.n_species
        )

        # update the mass fraction
        self.mass_fraction_left_upwind = calculate_multispecies_three_point_array(
            mass_fraction,
            surface.gas_surface.Y,
            gas_inf.Y,
            three_points_scheme.scheme_upwind_left,
            grid.gas_grid.cell_count,
            gas_array_iter.n_species
        )
        self.mass_fraction_right_upwind = calculate_multispecies_three_point_array(
            mass_fraction,
            surface.gas_surface.Y,
            gas_inf.Y,
            three_points_scheme.scheme_upwind_right,
            grid.gas_grid.cell_count,
            gas_array_iter.n_species
        )

        # calculate the mass fraction gradient
        self.mass_fraction_gradient_left, self.mass_fraction_gradient_right = calculate_species_gradients(
            mass_fraction, grid.gas_grid.positions_volume_centers,
            surface.gas_surface.Y, gas_inf.Y,
            grid.params.droplet_radius, grid.params.r_inf
        )

        # check the negative values in the mass fraction related arrays
        for name, array in {
            "density_left_upwind": self.density_left_upwind,
            "density_right_upwind": self.density_right_upwind,
            "rho_diffusivity_left_center": self.rho_diffusivity_left_center,
            "rho_diffusivity_right_center": self.rho_diffusivity_right_center,
            "mass_fraction_left_upwind": self.mass_fraction_left_upwind,
            "mass_fraction_right_upwind": self.mass_fraction_right_upwind
        }.items():
            if np.any(array < -1e-3):
                negative_indices = np.where(array < -1e-3)
                if len(array.shape) == 1:
                    indices_str = f"location: {negative_indices[0]}"
                else:
                    indices_str = f"location: {list(zip(*negative_indices))}"
                warnings.warn(f"negative values detected in {name}, minimum value: {np.min(array)}, {indices_str}")
        
        # check for 0 values in the diffusion coefficients
        self.check_rho_diffusivity_zeros()

    def liquid_update_equation_property(self, liquid_array_iter: 'LiquidArray', grid: Grid, liquid_00: 'Liquid', surface: 'Surface', three_points_scheme: ThreePointsScheme, mass_fraction: np.ndarray):
        """update the parameters related to the liquid phase equation
        
        Args:
            liquid_array_iter: liquid phase state array
            grid: grid object
            liquid_00: liquid phase state at the center of the droplet (left boundary)
            surface: surface object (right boundary)
            three_points_scheme: three points scheme coefficients
            mass_fraction: mass fraction array
        """
        # update the parameters related to temperature
        self.liquid_update_equation_property_T(liquid_array_iter, grid, liquid_00, surface, three_points_scheme)
        
        # update the parameters related to mass fraction
        self.liquid_update_equation_property_Y(liquid_array_iter, grid, liquid_00, surface, three_points_scheme, mass_fraction)

    def liquid_update_equation_property_T(self, liquid_array_iter: 'LiquidArray', grid: Grid, liquid_00: 'Liquid', surface: 'Surface', three_points_scheme: ThreePointsScheme):
        """update the parameters related to temperature in the liquid phase equation
        
        Args:
            liquid_array_iter: liquid phase state array
            grid: grid object
            liquid_00: liquid phase state at the center of the droplet (left boundary)
            surface: surface object (right boundary)
            three_points_scheme: three points scheme coefficients
        """
        # update the specific enthalpy
        self.specific_enthalpy_left_upwind = calculate_three_point_array(
            liquid_array_iter.cp_mass * liquid_array_iter.density_mass,
            liquid_00.cp_mass * liquid_00.density_mass,
            surface.liquid_surface.cp_mass * surface.liquid_surface.density_mass,
            three_points_scheme.scheme_upwind_left,
            grid.liquid_grid.cell_count
        )
        self.specific_enthalpy_right_upwind = calculate_three_point_array(
            liquid_array_iter.cp_mass * liquid_array_iter.density_mass,
            liquid_00.cp_mass * liquid_00.density_mass,
            surface.liquid_surface.cp_mass * surface.liquid_surface.density_mass,
            three_points_scheme.scheme_upwind_right,
            grid.liquid_grid.cell_count
        )

        # update the thermal conductivity
        self.thermal_conductivity_left_center = calculate_three_point_array(
            liquid_array_iter.thermal_conductivity,
            liquid_00.thermal_conductivity,
            surface.liquid_surface.thermal_conductivity,
            three_points_scheme.scheme_center_left,
            grid.liquid_grid.cell_count
        )
        self.thermal_conductivity_right_center = calculate_three_point_array(
            liquid_array_iter.thermal_conductivity,
            liquid_00.thermal_conductivity,
            surface.liquid_surface.thermal_conductivity,
            three_points_scheme.scheme_center_right,
            grid.liquid_grid.cell_count
        )

        # update the temperature
        self.temperature_left_upwind = calculate_three_point_array(
            liquid_array_iter.T,
            liquid_00.temperature,
            surface.liquid_surface.temperature,
            three_points_scheme.scheme_upwind_left,
            grid.liquid_grid.cell_count
        )
        self.temperature_right_upwind = calculate_three_point_array(
            liquid_array_iter.T,
            liquid_00.temperature,
            surface.liquid_surface.temperature,
            three_points_scheme.scheme_upwind_right,
            grid.liquid_grid.cell_count
        )

        # calculate the temperature gradient
        self.temperature_gradient_left, self.temperature_gradient_right = calculate_temperature_gradients(
            liquid_array_iter.T, grid.liquid_grid.positions_volume_centers,
            liquid_00.temperature, surface.liquid_surface.temperature,
            0.0, grid.params.droplet_radius  # the left boundary of the liquid phase is 0, the right boundary is the droplet radius
        )

        # check the negative values in the temperature related arrays
        for name, array in {
            "specific_enthalpy_left_upwind": self.specific_enthalpy_left_upwind,
            "specific_enthalpy_right_upwind": self.specific_enthalpy_right_upwind,
            "thermal_conductivity_left_center": self.thermal_conductivity_left_center,
            "thermal_conductivity_right_center": self.thermal_conductivity_right_center,
            "temperature_left_upwind": self.temperature_left_upwind,
            "temperature_right_upwind": self.temperature_right_upwind
        }.items():
            if np.any(array < 0):
                negative_indices = np.where(array < 0)
                indices_str = f"location: {negative_indices[0]}"
                warnings.warn(f"negative values detected in {name}, minimum value: {np.min(array)}, {indices_str}")

    def liquid_update_equation_property_Y(self, liquid_array_iter: 'LiquidArray', grid: Grid, liquid_00: 'Liquid', surface: 'Surface', three_points_scheme: ThreePointsScheme, mass_fraction: np.ndarray):
        """update the parameters related to mass fraction in the liquid phase equation
        
        Args:
            liquid_array_iter: liquid phase state array
            grid: grid object
            liquid_00: liquid phase state at the center of the droplet (left boundary)
            surface: surface object (right boundary)
            three_points_scheme: three points scheme coefficients
            mass_fraction: mass fraction array
        """
        # calculate the product of density and diffusion coefficient
        rho_diffusivity = liquid_array_iter.density_mass[:, np.newaxis] * liquid_array_iter.diffusion_mean
        rho_diffusivity_00 = liquid_00.density_mass * liquid_00.diffusion_mean
        rho_diffusivity_surface = surface.liquid_surface.density_mass * surface.liquid_surface.diffusion_mean

        # update the density
        self.density_left_upwind = calculate_three_point_array(
            liquid_array_iter.density_mass,
            liquid_00.density_mass,
            surface.liquid_surface.density_mass,
            three_points_scheme.scheme_upwind_left,
            grid.liquid_grid.cell_count
        )
        self.density_right_upwind = calculate_three_point_array(
            liquid_array_iter.density_mass,
            liquid_00.density_mass,
            surface.liquid_surface.density_mass,
            three_points_scheme.scheme_upwind_right,
            grid.liquid_grid.cell_count
        )

        # update the diffusion coefficient
        self.rho_diffusivity_left_center = calculate_multispecies_three_point_array(
            rho_diffusivity,
            rho_diffusivity_00,
            rho_diffusivity_surface,
            three_points_scheme.scheme_center_left,
            grid.liquid_grid.cell_count,
            len(liquid_array_iter.liquids[0].composition)
        )
        self.rho_diffusivity_right_center = calculate_multispecies_three_point_array(
            rho_diffusivity,
            rho_diffusivity_00,
            rho_diffusivity_surface,
            three_points_scheme.scheme_center_right,
            grid.liquid_grid.cell_count,
            len(liquid_array_iter.liquids[0].composition)
        )

        # update the mass fraction
        self.mass_fraction_left_upwind = calculate_multispecies_three_point_array(
            mass_fraction,
            liquid_00.mass_fraction,
            surface.liquid_surface.mass_fraction,
            three_points_scheme.scheme_upwind_left,
            grid.liquid_grid.cell_count,
            len(liquid_array_iter.liquids[0].composition)
        )
        self.mass_fraction_right_upwind = calculate_multispecies_three_point_array(
            mass_fraction,
            liquid_00.mass_fraction,
            surface.liquid_surface.mass_fraction,
            three_points_scheme.scheme_upwind_right,
            grid.liquid_grid.cell_count,
            len(liquid_array_iter.liquids[0].composition)
        )

        # calculate the mass fraction gradient
        self.mass_fraction_gradient_left, self.mass_fraction_gradient_right = calculate_species_gradients(
            mass_fraction, grid.liquid_grid.positions_volume_centers,
            liquid_00.mass_fraction, surface.liquid_surface.mass_fraction,
            0.0, grid.params.droplet_radius  # 液相的左边界是0，右边界是液滴半径
        )

        # check the negative values in the mass fraction related arrays
        for name, array in {
            "density_left_upwind": self.density_left_upwind,
            "density_right_upwind": self.density_right_upwind,
            "rho_diffusivity_left_center": self.rho_diffusivity_left_center,
            "rho_diffusivity_right_center": self.rho_diffusivity_right_center,
            "mass_fraction_left_upwind": self.mass_fraction_left_upwind,
            "mass_fraction_right_upwind": self.mass_fraction_right_upwind
        }.items():
            if np.any(array < 0):
                negative_indices = np.where(array < 0)
                if len(array.shape) == 1:
                    indices_str = f"位置: {negative_indices[0]}"
                else:
                    indices_str = f"位置: {list(zip(*negative_indices))}"
                warnings.warn(f"在 {name} 中检测到负值，最小值: {np.min(array)}，{indices_str}")
        
        # check for 0 values in the diffusion coefficients
        self.check_rho_diffusivity_zeros()



@jit(nopython=True, cache=True)
def check_arrays_min(arrays):
    return np.min(arrays)



@jit(nopython=True, cache=True)
def calculate_three_point_array(mid_values: np.ndarray, left_boundary_value: float, 
                          right_boundary_value: float,scheme_array: np.ndarray, n_cells: int) -> np.ndarray:
    """calculate the three points array"""
    three_point_array = create_three_point_array(mid_values, left_boundary_value, right_boundary_value, n_cells)
    return np.sum(three_point_array * scheme_array, axis=1)


@jit(nopython=True, cache=True)
def create_three_point_array(mid_values: np.ndarray, left_boundary_value: float, 
                          right_boundary_value: float, n_cells: int) -> np.ndarray:
    """create the three points array [left, middle, right]
    
    Args:
        mid_values: the values of the middle grid (current grid)
        left_boundary_value: the value of the left boundary
        right_boundary_value: the value of the right boundary
        n_cells: the number of cells
        
    Returns:
        np.ndarray: the three points array, shape is (n_cells, 3)
    """
    result = np.zeros((n_cells, 3))
    
    # the middle value (index 1)
    result[:, 1] = mid_values
    
    # the left value (index 0)
    result[0, 0] = left_boundary_value  # the first grid uses the left boundary value
    result[1:, 0] = mid_values[:-1]     # other grids use the value of the previous grid
    
    # the right value (index 2)
    result[:-1, 2] = mid_values[1:]     # other grids use the value of the next grid
    result[-1, 2] = right_boundary_value # the last grid uses the right boundary value
    
    return result

@jit(nopython=True, cache=True)
def create_multispecies_three_point_array(mid_values: np.ndarray, left_boundary_value: np.ndarray,
                                       right_boundary_value: np.ndarray, n_cells: int, n_species: int) -> np.ndarray:
    """create the three points array for multi-species physical parameters
    
    Args:
        mid_values: the values of the middle grid, shape is (n_cells, n_species)
        left_boundary_value: the value of the left boundary, shape is (n_species,)
        right_boundary_value: the value of the right boundary, shape is (n_species,)
        n_cells: the number of cells
        n_species: the number of species
        
    Returns:
        np.ndarray: the three points array, shape is (n_cells, 3, n_species)
    """
    # pre-allocate the result array
    result = np.empty((n_cells, 3, n_species))
    
    # the middle value (index 1)
    result[:, 1, :] = mid_values
    
    # the left value (index 0)
    result[1:, 0, :] = mid_values[:-1, :]
    result[0, 0, :] = left_boundary_value
    
    # the right value (index 2)
    result[:-1, 2, :] = mid_values[1:, :]
    result[-1, 2, :] = right_boundary_value
    
    return result

@jit(nopython=True, cache=True)
def calculate_multispecies_three_point_array(mid_values: np.ndarray, left_boundary_value: np.ndarray,
                                         right_boundary_value: np.ndarray, scheme_array: np.ndarray,
                                         n_cells: int, n_species: int) -> np.ndarray:
    """calculate the three points array for multi-species physical parameters
    
    Args:
        mid_values: the values of the middle grid, shape is (n_cells, n_species)
        left_boundary_value: the value of the left boundary, shape is (n_species,)
        right_boundary_value: the value of the right boundary, shape is (n_species,)
        scheme_array: the scheme array, shape is (n_cells, 3)
        n_cells: the number of cells
        n_species: the number of species
        
    Returns:
        np.ndarray: the result array, shape is (n_cells, n_species)
    """
    # create the three points array
    three_point_array = create_multispecies_three_point_array(
        mid_values, left_boundary_value, right_boundary_value, n_cells, n_species
    )
    
    # initialize the result array
    result = np.empty((n_cells, n_species), dtype=np.float64)
    
    # parallel calculation
    for i in range(n_cells):  
        for k in range(n_species):
            result[i, k] = np.sum(three_point_array[i, :, k] * scheme_array[i, :])
    
    return result

@jit(nopython=True, cache=True)
def calculate_temperature_gradients(temperature: np.ndarray, volume_centers: np.ndarray,
                                  surface_temp: float, gas_inf_temp: float,
                                  droplet_radius: float, r_inf: float) -> tuple[np.ndarray, np.ndarray]:
    """calculate the temperature gradient
    
    Args:
        temperature: the temperature array
        volume_centers: the volume center coordinates array
        surface_temp: the surface temperature
        gas_inf_temp: the infinite far temperature
        droplet_radius: the droplet radius
        r_inf: the infinite far radius
        
    Returns:
        tuple[np.ndarray, np.ndarray]: (left gradient, right gradient)
    """
    n = len(temperature)
    gradient_left = np.zeros(n)
    gradient_right = np.zeros(n)
    
    # calculate the temperature gradient of the internal grid
    gradient_left[1:] = (temperature[1:] - temperature[:-1]) / (volume_centers[1:] - volume_centers[:-1])
    gradient_right[:-1] = gradient_left[1:]
    
    # handle the temperature gradient of the boundary grid
    gradient_left[0] = (temperature[0] - surface_temp) / (volume_centers[0] - droplet_radius)
    gradient_right[-1] = (gas_inf_temp - temperature[-1]) / (r_inf - volume_centers[-1])
    
    return gradient_left, gradient_right

@jit(nopython=True, cache=True)
def calculate_species_gradients(mass_fraction: np.ndarray, volume_centers: np.ndarray,
                              surface_mass_fraction: np.ndarray, gas_inf_mass_fraction: np.ndarray,
                              droplet_radius: float, r_inf: float) -> tuple[np.ndarray, np.ndarray]:
    """calculate the mass fraction gradient
    
    Args:
        mass_fraction: the mass fraction array, shape is (n_cells, n_species)
        volume_centers: the volume center coordinates array
        surface_mass_fraction: the surface mass fraction array, shape is (n_species,)
        gas_inf_mass_fraction: the infinite far mass fraction array, shape is (n_species,)
        droplet_radius: the droplet radius
        r_inf: the infinite far radius
        
    Returns:
        tuple[np.ndarray, np.ndarray]: (left gradient, right gradient), shape is (n_cells, n_species)
    """
    n_cells, n_species = mass_fraction.shape
    gradient_left = np.zeros((n_cells, n_species))
    gradient_right = np.zeros((n_cells, n_species))
    
    # calculate the mass fraction gradient of the internal grid
    for i in range(1, n_cells):
        for j in range(n_species):
            gradient_left[i, j] = (mass_fraction[i, j] - mass_fraction[i-1, j]) / (volume_centers[i] - volume_centers[i-1])
            gradient_right[i-1, j] = gradient_left[i, j]
    
    # handle the mass fraction gradient of the boundary grid
    for j in range(n_species):
        gradient_left[0, j] = (mass_fraction[0, j] - surface_mass_fraction[j]) / (volume_centers[0] - droplet_radius)
        gradient_right[-1, j] = (gas_inf_mass_fraction[j] - mass_fraction[-1, j]) / (r_inf - volume_centers[-1])
    
    return gradient_left, gradient_right