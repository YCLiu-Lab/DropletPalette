"""
surface module

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.


This module contains two main classes:
1. SurfaceState: surface state data class
2. Surface: surface management class

main functions:
- surface state management
- gas-liquid interface mass and heat transfer
- component balance calculation
"""

import numpy as np
import cantera as ct
from dataclasses import dataclass, field
from .grid import Grid
from src.solution import init_species_mapping, Liquid

@dataclass
class SurfaceState:
    """surface state data class
    
    used to store and manage the state parameters of the gas-liquid interface
    
    Attributes:
        temperature: temperature [K]
        stefan_velocity_liquid: liquid phase Stefan velocity [m/s]
        stefan_velocity_gas: gas phase Stefan velocity [m/s]
        old_stefan_velocity_liquid: previous time liquid phase Stefan velocity [m/s]
        old_stefan_velocity_gas: previous time gas phase Stefan velocity [m/s]
        interface_diffusion_flux_liquid: liquid phase fuel interface diffusion flux [kg/m^2/s]
        interface_diffusion_flux_gas: gas phase fuel interface diffusion flux [kg/m^2/s]
        evaporation_rate: evaporation rate [kg/s]
        heat_flux_liquid: liquid phase heat transfer rate [W/m^2]
        heat_flux_gas: gas phase heat transfer rate [W/m^2]
        evaporation_heat_flux: evaporation heat transfer rate [W/m^2]
    """
    temperature: float = 0.0
    stefan_velocity_liquid: float = 0.0
    stefan_velocity_gas: float = 0.0
    old_stefan_velocity_liquid: float = 0.0
    old_stefan_velocity_gas: float = 0.0
    interface_diffusion_flux_liquid: np.ndarray = field(default_factory=lambda: np.zeros(40))
    interface_diffusion_flux_gas: np.ndarray = field(default_factory=lambda: np.zeros(40))
    evaporation_rate:  np.ndarray = field(default_factory=lambda: np.zeros(40))
    heat_flux_liquid: float = 0.0
    heat_flux_gas: float = 0.0
    evaporation_heat_flux: float = 0.0

class Surface:
    """surface management class
    
    used to manage the state parameters and boundary conditions of the droplet surface, including:
    1. gas phase surface state
    2. liquid phase surface state
    3. surface temperature
    4. surface heat flux
    5. surface mass flux
    """
    
    def __init__(self, grid: Grid, gas_surface: ct.Solution, liquid_surface: 'Liquid'):
        """initialize the surface object
        
        Args:
            grid: grid object
            gas_surface: gas phase surface state
            liquid_surface: liquid phase surface state
        """
        self.grid = grid
        self.gas_surface = gas_surface
        self.liquid_surface = liquid_surface
        self.temperature = None
        self.heat_flux = None
        self.mass_flux = None
        self.state = SurfaceState(temperature=self.liquid_surface.temperature)
        self._precompute_indices()
        self.initialize_gas_surface_state()

    def _precompute_indices(self):
        """
        precompute and cache the commonly used component index mapping.
        
        mainly includes:
            - gas phase fuel component index (gas_fuel_indices)
            - gas phase non-fuel component index (gas_non_fuel_indices)
            - liquid phase component index with corresponding gas phase component (liquid_fuel_indices)
            - liquid phase component index without corresponding gas phase component (liquid_non_fuel_indices)
            - liquid phase to gas phase component mapping (species_liquid2gas)
        """
        self.species_names_gas = self.gas_surface.species_names
        self.species_names_liquid = self.liquid_surface.species_names
        self.species_liquid2gas = init_species_mapping(self.species_names_gas)
        self.gas_fuel_indices = np.array([v for v in self.species_liquid2gas.values() if v is not None])
        self.gas_non_fuel_indices = np.array([i for i in range(len(self.species_names_gas)) 
                                        if i not in self.gas_fuel_indices])
        self.liquid_fuel_indices = np.array([i for i, v in self.species_liquid2gas.items() if v is not None])
        self.liquid_non_fuel_indices = np.array([i for i, v in self.species_liquid2gas.items() if v is None])

    def initialize_gas_surface_state(self):
        """
        initialize the gas phase surface state.
        set the gas phase surface state to be consistent with the liquid phase surface temperature, pressure and component balance.
        """
        self.gas_surface.TPX = (
            self.liquid_surface.temperature,
            self.liquid_surface.pressure,
            self.calculate_gas_surface_composition()
        )

    def calculate_gas_surface_composition(self):
        """
        calculate the gas-liquid interface component balance, return the normalized gas phase component mole fraction.
        
        Returns:
            np.ndarray: gas phase surface component mole fraction (normalized)
        """
        gas_mole_fractions = np.array(self.gas_surface.X)
        fuel_partial_pressure = np.zeros_like(gas_mole_fractions)
        vapor_pressure = self.liquid_surface.vapor_pressure_ij()
        activity_coefficient = self.liquid_surface.activity_coefficient
        composition_liquid = self.liquid_surface.composition

        # only assign values to components with mapping
        fuel_partial_pressure[self.gas_fuel_indices] = (
            vapor_pressure[self.liquid_fuel_indices] *
            activity_coefficient[self.liquid_fuel_indices] *
            composition_liquid[self.liquid_fuel_indices]
        )
        gas_mole_fractions[self.gas_fuel_indices] = fuel_partial_pressure[self.gas_fuel_indices] / self.gas_surface.P

        # non-fuel component normalization
        fuel_mole_fraction_sum = np.sum(gas_mole_fractions[self.gas_fuel_indices])
        non_fuel_mole_fraction_total = 1.0 - fuel_mole_fraction_sum
        if non_fuel_mole_fraction_total > 0:
            current_non_fuel_total = np.sum(gas_mole_fractions[self.gas_non_fuel_indices])
            if current_non_fuel_total > 0:
                scale_factor = non_fuel_mole_fraction_total / current_non_fuel_total
                gas_mole_fractions[self.gas_non_fuel_indices] *= scale_factor
        # final normalization
        return gas_mole_fractions / np.sum(gas_mole_fractions)
