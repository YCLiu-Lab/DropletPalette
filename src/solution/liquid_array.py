"""
liquid array module

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.
"""


import numpy as np
from typing import Dict, List, Optional
from .liquid import Liquid

class LiquidArray:
    """liquid phase state sequence class, used to manage multiple liquid phase states
    
    this class is used to manage multiple liquid phase states in one dimension, each state point contains temperature, pressure and component information.
    provides convenient attribute access to obtain physical properties at each location, supports two types of component representation: mole fraction and mass fraction.
    
    main functions:
    1. state management:
       - support setting state through TPX(temperature, pressure, mole fraction) and TPY(temperature, pressure, mass fraction)
       - support setting state for single location or all locations
       - provide attribute access to temperature, pressure, mole fraction and mass fraction
    
    2. physical property calculation:
       - average molecular weight of mixture (kg/kmol)
       - molar density and mass density (kmol/m³, kg/m³)
       - molar heat capacity and mass heat capacity (J/kmol·K, J/kg·K)
       - molar heat of vaporization and mass heat of vaporization (J/kmol, J/kg)
       - thermal conductivity (W/m·K)
       - saturated vapor pressure (Pa)
       - viscosity (Pa·s)
       - average diffusion coefficient (m²/s)
       - activity coefficient
    
    3. data access:
       - support getting all physical properties at a single location
       - support getting basic physical properties at all locations
       - support getting all physical properties at all locations
       - support getting a specific physical property at all locations
    
    attributes:
        n_grid (int): number of liquid phase grids
        liquids (List[Liquid]): liquid phase state object list
    """
    
    def __init__(self, liquid_template: Liquid, n_grid: int):
        """initialize the liquid phase state sequence
        
        Args:
            liquid_template: liquid phase template object, used to create new Liquid objects
            n_grid: number of liquid phase grids
        """
        self.n_grid = n_grid
        self.liquids: List[Liquid] = []
        
        # use template to create n_grid Liquid objects
        for _ in range(n_grid):
            new_liquid = Liquid(
                temperature=liquid_template.temperature,
                pressure=liquid_template.pressure,
                composition=liquid_template.composition.copy()
            )
            self.liquids.append(new_liquid)
            
    @property
    def T(self) -> np.ndarray:
        """get the temperature at all locations
        
        Returns:
            np.ndarray: array of shape (n,), representing the temperature values at n locations, unit: K
        """
        return np.array([liquid.temperature for liquid in self.liquids])
        
    @property
    def P(self) -> np.ndarray:
        """get the pressure at all locations
        
        Returns:
            np.ndarray: array of shape (n,), representing the pressure values at n locations, unit: Pa
        """
        return np.array([liquid.pressure for liquid in self.liquids])
        
    @property
    def X(self) -> np.ndarray:
        """get the mole fraction at all locations
        
        Returns:
            np.ndarray: array of shape (n,40), representing the mole fraction values at n locations, unit: kmol/kmol
        """
        return np.array([liquid.composition for liquid in self.liquids])
        
    @property
    def Y(self) -> np.ndarray:
        """get the mass fraction at all locations
        
        Returns:
            np.ndarray: array of shape (n,40), representing the mass fraction values at n locations, unit: kg/kg
        """
        return np.array([liquid.mass_fraction for liquid in self.liquids])
        

    def TPX(self, temperature: Optional[np.ndarray] = None, 
            pressure: Optional[np.ndarray] = None,
            composition: Optional[np.ndarray] = None) -> None:
        """set the temperature, pressure and mole fraction at all locations
        
        Args:
            temperature: array of shape (n,), representing the temperature values at n locations, unit: K
            pressure: array of shape (n,), representing the pressure values at n locations, unit: Pa
            composition: array of shape (n,40), representing the mole fraction values at n locations, unit: kmol/kmol
        """
        # use the TPY method of each Liquid object to set the state
        for i, liquid in enumerate(self.liquids):
            liquid.TPX(
                temperature=temperature[i] if temperature is not None else None,
                pressure=pressure[i] if pressure is not None else None,
                composition=composition[i] if composition is not None else None
            )
                
    def TPY(self, temperature: Optional[np.ndarray] = None, 
            pressure: Optional[np.ndarray] = None,
            mass_fraction: Optional[np.ndarray] = None) -> None:
        """set the temperature, pressure and mass fraction at all locations
        
        Args:
            temperature: array of shape (n,), representing the temperature values at n locations, unit: K
            pressure: array of shape (n,), representing the pressure values at n locations, unit: Pa
            mass_fraction: array of shape (n,40), representing the mass fraction values at n locations, unit: kg/kg
        """
        # use the TPY method of each Liquid object to set the state
        for i, liquid in enumerate(self.liquids):
            liquid.TPY(
                temperature=temperature[i] if temperature is not None else None,
                pressure=pressure[i] if pressure is not None else None,
                mass_fraction=mass_fraction[i] if mass_fraction is not None else None
            )

    def TPX_i(self, loc: int,
              temperature: Optional[float] = None,
              pressure: Optional[float] = None,
              composition: Optional[np.ndarray] = None) -> None:
        """set the temperature, pressure and mole fraction at a specific location
        
        Args:
            loc: location index
            temperature: temperature value, unit: K
            pressure: pressure value, unit: Pa
            composition: array of shape (40,), representing the mole fraction values at 40 components
        """
        if loc < 0 or loc >= self.n_grid:
            raise ValueError(f"location index {loc} out of range [0, {self.n_grid-1}]")
            
        # use the TPY method of the Liquid object to set the state
        self.liquids[loc].TPX(
            temperature=temperature,
            pressure=pressure,
            composition=composition
        )
            
    def TPY_i(self, loc: int,
              temperature: Optional[float] = None,
              pressure: Optional[float] = None,
              mass_fraction: Optional[np.ndarray] = None) -> None:
        """set the temperature, pressure and mass fraction at a specific location
        
        Args:
            loc: location index
            temperature: temperature value, unit: K
            pressure: pressure value, unit: Pa
            mass_fraction: array of shape (40,), representing the mass fraction values at 40 components
        """
        if loc < 0 or loc >= self.n_grid:
            raise ValueError(f"location index {loc} out of range [0, {self.n_grid-1}]")
            
        # use the TPY method of the Liquid object to set the state
        self.liquids[loc].TPY(
            temperature=temperature,
            pressure=pressure,
            mass_fraction=mass_fraction
        )
        
    @property
    def molecular_weight(self) -> np.ndarray:
        """get the average molecular weight of the mixture at all locations
        
        Returns:
            np.ndarray: array of shape (n,), representing the average molecular weight of the mixture at n locations, unit: kg/kmol
        """
        return self.get_single_property('molecular_weight')
        
    @property
    def density_mole(self) -> np.ndarray:
        """get the molar density at all locations
        
        Returns:
            np.ndarray: array of shape (n,), representing the molar density at n locations, unit: mol/m³
        """
        return self.get_single_property('density_mole')
        
    @property
    def density_mass(self) -> np.ndarray:
        """get the mass density at all locations
        
        Returns:
            np.ndarray: array of shape (n,), representing the mass density at n locations, unit: kg/m³
        """
        return self.get_single_property('density_mass')
        
    @property
    def cp_mole(self) -> np.ndarray:
        """get the molar heat capacity at all locations
        
        Returns:
            np.ndarray: array of shape (n,), representing the molar heat capacity at n locations, unit: J/kmol·K
        """
        return self.get_single_property('cp_mole')
        
    @property
    def cp_mass(self) -> np.ndarray:
        """get the mass heat capacity at all locations
        
        Returns:
            np.ndarray: array of shape (n,), representing the mass heat capacity at n locations, unit: J/kg·K
        """
        return self.get_single_property('cp_mass')
        
    @property
    def heat_vaporization_mole(self) -> np.ndarray:
        """get the molar heat of vaporization at all locations
        
        Returns:
            np.ndarray: array of shape (n,), representing the molar heat of vaporization at n locations, unit: J/kmol
        """
        return self.get_single_property('heat_vaporization_mole')
        
    @property
    def heat_vaporization_mass(self) -> np.ndarray:
        """get the mass heat of vaporization at all locations
        
        Returns:
            np.ndarray: array of shape (n,), representing the mass heat of vaporization at n locations, unit: J/kg
        """
        return self.get_single_property('heat_vaporization_mass')
        
    @property
    def thermal_conductivity(self) -> np.ndarray:
        """get the thermal conductivity at all locations
        
        Returns:
            np.ndarray: array of shape (n,), representing the thermal conductivity at n locations, unit: W/m·K
        """
        return self.get_single_property('thermal_conductivity')
        
    @property
    def vapor_pressure(self) -> np.ndarray:
        """get the saturated vapor pressure at all locations
        
        Returns:
            np.ndarray: array of shape (n,), representing the saturated vapor pressure at n locations, unit: Pa
        """
        return self.get_single_property('vapor_pressure')
        
    @property
    def viscosity(self) -> np.ndarray:
        """get the viscosity at all locations
        
        Returns:
            np.ndarray: array of shape (n,), representing the viscosity at n locations, unit: Pa·s
        """
        return self.get_single_property('viscosity')
        
    @property
    def diffusion_mean(self) -> np.ndarray:
        """get the average diffusion coefficient at all locations
        
        Returns:
            np.ndarray: array of shape (n,40), representing the average diffusion coefficient at n locations, unit: m²/s
        """
        return self.get_single_property('diffusion_mean')
        
    @property
    def activity_coefficient(self) -> np.ndarray:
        """get the activity coefficient at all locations
        
        Returns:
            np.ndarray: array of shape (n,40), representing the activity coefficient at n locations, unit: -
        """
        return self.get_single_property('activity_coefficient')
    

    def get_property_at_location(self, loc: int) -> Dict[str, float]:
        """get the basic physical properties at a specific location
        
        Args:
            loc: location index
            
        Returns:
            Dict[str, float]: dictionary containing the following basic physical properties:
                - density: mass density (kg/m³)
                - viscosity: viscosity (Pa·s)
                - thermal_conductivity: thermal conductivity (W/m·K)
                - heat_capacity: mass heat capacity (J/kg·K)
                - diffusion: average diffusion coefficient (m²/s)
        """
        if loc < 0 or loc >= self.n_grid:
            raise ValueError(f"location index {loc} out of range [0, {self.n_grid-1}]")
            
        liquid = self.liquids[loc]
        return {
            'density': liquid.density_mass,
            'viscosity': liquid.viscosity,
            'thermal_conductivity': liquid.thermal_conductivity,
            'heat_capacity': liquid.cp_mass,
            'diffusion': liquid.diffusion_mean
        }
        
    def get_all_basic_properties(self) -> Dict[str, np.ndarray]:
        """get the basic physical properties at all locations
        
        Returns:
            Dict[str, np.ndarray]: dictionary containing the following basic physical properties:
                - density: mass density (kg/m³)
                - viscosity: viscosity (Pa·s)
                - thermal_conductivity: thermal conductivity (W/m·K)
                - heat_capacity: mass heat capacity (J/kg·K)
                - diffusion: average diffusion coefficient (m²/s)
        """
        return {
            'density': self.density_mass,
            'viscosity': self.viscosity,
            'thermal_conductivity': self.thermal_conductivity,
            'heat_capacity': self.cp_mass,
            'diffusion': self.diffusion_mean
        }
        
    def get_all_physical_properties(self) -> Dict[str, np.ndarray]:
        """get the complete physical properties at all locations
        
        Returns:
            Dict[str, np.ndarray]: dictionary containing the following physical properties:
                - molecular_weight: average molecular weight of the mixture (kg/kmol)
                - density_mole: molar density (kmol/m³)
                - density_mass: mass density (kg/m³)
                - cp_mole: molar heat capacity (J/kmol·K)
                - cp_mass: mass heat capacity (J/kg·K)
                - heat_vaporization_mole: molar heat of vaporization (J/kmol)
                - heat_vaporization_mass: mass heat of vaporization (J/kg)
                - thermal_conductivity: thermal conductivity (W/m·K)
                - vapor_pressure: saturated vapor pressure (Pa)
                - viscosity: viscosity (Pa·s)
                - diffusion_mean: average diffusion coefficient (m²/s)
                - activity_coefficient: activity coefficient
        """
        return {
            'molecular_weight': self.molecular_weight,
            'density_mole': self.density_mole,
            'density_mass': self.density_mass,
            'cp_mole': self.cp_mole,
            'cp_mass': self.cp_mass,
            'heat_vaporization_mole': self.heat_vaporization_mole,
            'heat_vaporization_mass': self.heat_vaporization_mass,
            'thermal_conductivity': self.thermal_conductivity,
            'vapor_pressure': self.vapor_pressure,
            'viscosity': self.viscosity,
            'diffusion_mean': self.diffusion_mean,
            'activity_coefficient': self.activity_coefficient
        }

    def get_single_property(self, property_name: str) -> np.ndarray:
        """get the specific physical property at all locations
        
        Args:
            property_name: physical property name, optional values include:
                - molecular_weight: average molecular weight of the mixture (kg/kmol)
                - density_mole: molar density (kmol/m³)
                - density_mass: mass density (kg/m³)
                - cp_mole: molar heat capacity (J/kmol·K)
                - cp_mass: mass heat capacity (J/kg·K)
                - heat_vaporization_mole: molar heat of vaporization (J/kmol)
                - heat_vaporization_mass: mass heat of vaporization (J/kg)
                - thermal_conductivity: thermal conductivity (W/m·K)
                - vapor_pressure: saturated vapor pressure (Pa)
                - viscosity: viscosity (Pa·s)
                - diffusion_mean: average diffusion coefficient (m²/s)
                - activity_coefficient: activity coefficient
                
        Returns:
            np.ndarray: array of shape (n,), representing the physical property values at n locations
            
        Exception:
            ValueError: when the physical property name is invalid, raise ValueError
        """
        # define the shape of the physical property
        property_shapes = {
            'molecular_weight': (self.n_grid,),
            'density_mole': (self.n_grid,),
            'density_mass': (self.n_grid,),
            'cp_mole': (self.n_grid,),
            'cp_mass': (self.n_grid,),
            'heat_vaporization_mole': (self.n_grid,),
            'heat_vaporization_mass': (self.n_grid,),
            'thermal_conductivity': (self.n_grid,),
            'vapor_pressure': (self.n_grid,),
            'viscosity': (self.n_grid,),
            'diffusion_mean': (self.n_grid,40),
            'activity_coefficient': (self.n_grid, 40)
        }
        
        # verify the physical property name
        if property_name not in property_shapes:
            raise ValueError(f"invalid physical property name: {property_name}")
            
        # create the result array
        result = np.zeros(property_shapes[property_name])
        
        # use vectorized operation to get the physical property
        if property_name in ['molecular_weight', 'density_mole', 'density_mass', 
                           'cp_mole', 'cp_mass', 'heat_vaporization_mole', 
                           'heat_vaporization_mass', 'thermal_conductivity', 
                           'vapor_pressure', 'viscosity']:
            # for one-dimensional array properties, use list comprehension
            result[:] = [getattr(liquid, property_name) for liquid in self.liquids]
        else:
            # for two-dimensional array properties, use list comprehension
            result[:] = [getattr(liquid, property_name) for liquid in self.liquids]
                
        return result

    def __getitem__(self, index: int) -> Liquid:
        """access the Liquid object at a specific location through index
        
        Args:
            index: location index
            
        Returns:
            Liquid: the Liquid object at the specified location
            
        Exception:
            IndexError: when the index is out of range, raise IndexError
        """
        if index < 0:
            index = len(self.liquids) + index
        if not 0 <= index < len(self.liquids):
            raise IndexError(f"index {index} out of range [0, {len(self.liquids)-1}]")
        return self.liquids[index]
