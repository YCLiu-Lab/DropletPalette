"""
Liquid solution class for droplet combustion simulation

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.
"""

from typing import TYPE_CHECKING, Optional, Union
import numpy as np
from . import liquid_para
from .liquid_para import MOLECULAR_WEIGHTS, SPECIES_NAMES
from .liquid_utils import validate_composition, validate_pressure, validate_temperature, normalize, find_supercritical_species, get_matrix_result
from .mapping_utils import LIQUID_SPECIES_NAMES

class Liquid:
    """liquid solution class for droplet combustion simulation
    
    this class is used to calculate the physical properties of liquid mixtures, including:
    1. basic state parameters:
       - temperature (K): temperature attribute
       - pressure (Pa): pressure attribute
       - composition: composition attribute
       - mass_fraction: mass_fraction attribute
    
    2. single component physical properties:
       - molar density (kmol/m³): density_mole_ij(i) method
       - mass density (kg/m³): density_mass_ij(i) method
       - molar heat capacity (J/kmol·K): cp_mole_ij(i) method
       - mass heat capacity (J/kg·K): cp_mass_ij(i) method
       - molar heat of vaporization (J/kmol): heat_vaporization_mole_ij(i) method
       - mass heat of vaporization (J/kg): heat_vaporization_mass_ij(i) method
       - thermal conductivity (W/m·K): thermal_conductivity_ij(i) method
       - saturated vapor pressure (Pa): vapor_pressure_ij(i) method
       - viscosity (Pa·s): viscosity_ij(i) method
       - diffusion coefficient (m²/s): diffusion_ij(i1,j1,i2,j2) method
       where i is the component index, i1,j1,i2,j2 are the four-dimensional indices of the diffusion coefficient
    
    3. mixture physical properties:
       - average molecular weight (kg/kmol): molecular_weight attribute
       - molar density (kmol/m³): density_mole attribute
       - mass density (kg/m³): density_mass attribute
       - molar heat capacity (J/kmol·K): cp_mole attribute
       - mass heat capacity (J/kg·K): cp_mass attribute
       - molar heat of vaporization (J/kmol): heat_vaporization_mole attribute
       - mass heat of vaporization (J/kg): heat_vaporization_mass attribute
       - thermal conductivity (W/m·K): thermal_conductivity attribute
       - saturated vapor pressure (Pa): vapor_pressure attribute
       - viscosity (Pa·s): viscosity attribute
       - activity coefficient: activity_coefficient attribute
       - average diffusion coefficient (m²/s): diffusion_mean attribute
    
    characteristics:
    1. state update:
       - support updating temperature, pressure and mole fraction through TPX method
       - support updating temperature, pressure and mass fraction through TPY method
       - automatically normalize the composition
       - automatically check the supercritical state
    
    2. performance optimization:
       - use cache mechanism to avoid repeated calculations
       - temperature-related calculations are only updated when the temperature changes
       - mixture physical properties are updated when the temperature, pressure, and composition change
    
    3. data verification:
       - verify the temperature range
       - verify the pressure range
       - verify the composition array
       - check the supercritical state
    
    4. calculation characteristics:
       - support 40-component mixtures
       - consider the interaction between components
    """

    def __init__(self, temperature: float, pressure: float, composition: np.ndarray):
        """
        initialize the liquid solution
        
        Args:
            temperature (float): temperature, unit: K
            pressure (float): pressure, unit: Pa
            composition (np.ndarray): composition array, shape: (40,), representing the mole fraction of 40 components
        """        
        # verify the input parameters
        validate_temperature(temperature)
        validate_pressure(pressure)
        validate_composition(composition)
        
        # initialize the basic state parameters
        self._temperature = temperature
        self._pressure = pressure
        self._composition = composition.copy()
        
        # calculate the mass fraction
        mass = self.composition * MOLECULAR_WEIGHTS
        self._mass_fraction = normalize(mass)
        
        # check the supercritical state
        find_supercritical_species(self._composition, self._temperature)
        
        # precompute the temperature-related parameters
        self._precompute_values()
        
        # initialize the cache
        self._clear_mixture_cache()
        self._clear_pure_cache()

    def _clear_mixture_cache(self):
        """clear the mixture physical property cache
        
        clear all the mixture physical property cache related to temperature, pressure and composition, including:
        - density (mole/mass)
        - heat capacity (mole/mass)
        - heat of vaporization (mole/mass)
        - thermal conductivity
        - saturated vapor pressure
        - viscosity
        - molecular weight
        - activity coefficient
        - average diffusion coefficient
        """
        # clear the mixture physical property cache (related to temperature, pressure and composition)
        self._cached_density_mole = None
        self._cached_density_mass = None
        self._cached_cp_mole = None
        self._cached_cp_mass = None
        self._cached_heat_vaporization_mole = None
        self._cached_heat_vaporization_mass = None
        self._cached_thermal_conductivity = None
        self._cached_vapor_pressure = None
        self._cached_viscosity = None
        self._cached_molecular_weight = None
        self._cached_activity_coefficient = None
        self._cached_diffusion_mean = None

    def _clear_pure_cache(self):    
        """clear the cache related to temperature
        
        clear all the pure physical property cache related to temperature, including:
        - density matrix (mole/mass)
        - heat capacity matrix (mole/mass)
        - heat of vaporization matrix (mole/mass)
        - thermal conductivity matrix
        - saturated vapor pressure matrix
        - viscosity matrix
        - diffusion coefficient matrix
        """
        # clear the pure physical property cache related to temperature
        self._cached_density_mole_ij = None
        self._cached_density_mass_ij = None
        self._cached_cp_mole_ij = None
        self._cached_cp_mass_ij = None
        self._cached_heat_vaporization_mole_ij = None
        self._cached_heat_vaporization_mass_ij = None
        self._cached_thermal_conductivity_ij = None
        self._cached_vapor_pressure_ij = None
        self._cached_viscosity_ij = None
        self._cached_diffusion_ij = None

    def _precompute_values(self):
        """precompute the temperature-related parameters
        
        calculate and store the temperature-related intermediate parameters, used for subsequent physical property calculations:
        - reduced temperature (Tr)
        - temperature parameters (tao, tao_035, tao_2, tao_3)
        - temperature power (T, T2, T3)
        - logarithmic term (ln_1_minus_Tr)
        """
        # call the precompute function in liquid_para
        (self._Tr, self._tao, self._tao_035, self._tao_2, 
         self._tao_3, self._T, self._T2, self._T3,
         self._ln_1_minus_Tr) = liquid_para.calculate_precompute_values(self._temperature)

    def __str__(self) -> str:
        """return the string representation of the state information
        
        返回:
            str: string representation of the state information, including temperature, pressure and non-zero component mole fractions
        """
        # only show the non-zero component mole fractions
        composition_str = {SPECIES_NAMES[i]: x for i, x in enumerate(self._composition) if x > 0}
        return f"state: temperature={self._temperature}K, pressure={self._pressure}Pa, composition={composition_str}"
        

    # 2. class state characteristic update and get method
    def TPX(self, temperature: float, pressure: float, composition: np.ndarray):
        """update the temperature, pressure and mole fraction
        
        calculation formula:
        - mole fraction normalization: xi = ni/Σni
        - mass fraction: wi = xi*Mi/Σ(xi*Mi)
        
        Args:
            temperature: temperature (K)
            pressure: pressure (Pa)
            composition: mole fraction array, shape: (40,)
        """
        # set the temperature and precompute
        self.temperature = temperature
        # set the pressure
        self.pressure = pressure
        # set the mole fraction and automatically update the mass fraction
        self.composition = composition        
        # check the supercritical species
        find_supercritical_species(self._composition, self._temperature)
        
    def TPY(self, temperature: float, pressure: float, mass_fraction: np.ndarray):
        """update the temperature, pressure and mass fraction
        
        calculation formula:
        - mole fraction: xi = (wi/Mi)/Σ(wi/Mi)
        - mass fraction normalization: wi = mi/Σmi
        
        Args:
            temperature: temperature (K)
            pressure: pressure (Pa)
            mass_fraction: mass fraction array, shape: (40,)
        """
        # set the temperature and precompute
        self.temperature = temperature
        # set the pressure
        self.pressure = pressure
        # set the mass fraction and automatically update the mole fraction
        self.mass_fraction = mass_fraction
        # check the supercritical species
        find_supercritical_species(self._composition, self._temperature)

    # property method
    @property
    def composition(self) -> np.ndarray:
        """get the component mole fraction
        
        Returns:
            np.ndarray: array of shape (40,), representing the mole fraction of 40 components
        """
        return self._composition.copy()
        
    @composition.setter
    def composition(self, value: np.ndarray):
        """set the component mole fraction
        
        Args:
            value: mole fraction array, shape: (40,)
        """
        validate_composition(value)
        if not np.array_equal(value, self._composition):  # only update when the composition changes
            self._composition = normalize(value)
            # update the mass fraction
            mass = self._composition * MOLECULAR_WEIGHTS
            self._mass_fraction = normalize(mass)
            # clear the mixture physical property cache
            self._clear_mixture_cache()

    @property
    def pressure(self) -> float:
        """get the pressure
        
        Returns:
            float: pressure value, unit: Pa
        """
        return self._pressure
        
    @pressure.setter
    def pressure(self, value: float):
        """set the pressure
        
        Args:
            value: pressure value, unit: Pa
        """
        validate_pressure(value)
        if value != self._pressure:  # only update when the pressure changes
            self._pressure = float(value)
            # clear the mixture physical property cache
            # TODO: IF we consider the properties depending on pressure, we need to change this part
            self._clear_mixture_cache()

    @property
    def temperature(self) -> float:
        """get the temperature
        
        Returns:
            float: temperature value, unit: K
        """
        return self._temperature
        
    @temperature.setter
    def temperature(self, value: float):
        """set the temperature
        
        Args:
            value: temperature value, unit: K
        """
        validate_temperature(value)        
        if value != self._temperature:  # only update when the temperature changes
            self._temperature = value
            self._precompute_values()
            # clear all the cache related to temperature
            self._clear_pure_cache()
            # clear the mixture physical property cache
            self._clear_mixture_cache()

    @property
    def mass_fraction(self) -> np.ndarray:
        """get the mass fraction
        
        Returns:
            np.ndarray: array of shape (40,), representing the mass fraction of 40 components
        """
        return self._mass_fraction.copy()
        
    @mass_fraction.setter
    def mass_fraction(self, value: np.ndarray):
        """set the mass fraction
        
        Args:
            value: mass fraction array, shape: (40,)
        """
        validate_composition(value)
        if not np.array_equal(value, self._mass_fraction):  # only update when the mass fraction changes
            # calculate the mole fraction and set
            moles = value / MOLECULAR_WEIGHTS
            self.composition = normalize(moles)
            # clear the mixture physical property cache
            self._clear_mixture_cache()

    # 3. single component physical property calculation - actual calculation process
    @property
    def _density_mole_ij(self) -> np.ndarray:
        """calculate the mole density matrix of single component"""
        if self._cached_density_mole_ij is None:
            self._cached_density_mole_ij = liquid_para.calculate_density_mole(
                self._tao, self._tao_035, self._tao_2, self._tao_3)
        return self._cached_density_mole_ij
        
    @property
    def _density_mass_ij(self) -> np.ndarray:
        """calculate the mass density matrix of single component"""
        if self._cached_density_mass_ij is None:
            self._cached_density_mass_ij = self._density_mole_ij * MOLECULAR_WEIGHTS
        return self._cached_density_mass_ij
        
    @property
    def _cp_mole_ij(self) -> np.ndarray:
        """calculate the mole heat capacity matrix of single component"""
        if self._cached_cp_mole_ij is None:
            self._cached_cp_mole_ij = liquid_para.calculate_cp_mole(
                self._tao, self._T, self._T2, self._T3)
        return self._cached_cp_mole_ij
        
    @property
    def _cp_mass_ij(self) -> np.ndarray:
        """calculate the mass heat capacity matrix of single component"""
        if self._cached_cp_mass_ij is None:
            self._cached_cp_mass_ij = self._cp_mole_ij / MOLECULAR_WEIGHTS
        return self._cached_cp_mass_ij
        
    @property
    def _heat_vaporization_mole_ij(self) -> np.ndarray:
        """calculate the mole heat of vaporization matrix of single component"""
        if self._cached_heat_vaporization_mole_ij is None:
            self._cached_heat_vaporization_mole_ij = liquid_para.calculate_heat_vaporization_mole(
                self._Tr, self._ln_1_minus_Tr)
        return self._cached_heat_vaporization_mole_ij
        
    @property
    def _heat_vaporization_mass_ij(self) -> np.ndarray:
        """calculate the mass heat of vaporization matrix of single component"""
        if self._cached_heat_vaporization_mass_ij is None:
            self._cached_heat_vaporization_mass_ij = self._heat_vaporization_mole_ij / MOLECULAR_WEIGHTS
        return self._cached_heat_vaporization_mass_ij
        
    @property
    def _thermal_conductivity_ij(self) -> np.ndarray:
        """calculate the thermal conductivity matrix of single component"""
        if self._cached_thermal_conductivity_ij is None:
            self._cached_thermal_conductivity_ij = liquid_para.calculate_thermal_conductivity(
                self._T, self._T2, self._T3)
        return self._cached_thermal_conductivity_ij
        
    @property
    def _vapor_pressure_ij(self) -> np.ndarray:
        """calculate the saturated vapor pressure matrix of single component"""
        if self._cached_vapor_pressure_ij is None:
            self._cached_vapor_pressure_ij = liquid_para.calculate_vapor_pressure(
                self._Tr, self._tao)
        return self._cached_vapor_pressure_ij
        
    @property
    def _viscosity_ij(self) -> np.ndarray:
        """calculate the viscosity matrix of single component"""
        if self._cached_viscosity_ij is None:
            self._cached_viscosity_ij = liquid_para.calculate_viscosity(
                self._T, self._T2, self._T3)
        return self._cached_viscosity_ij

    @property
    def _diffusion_ij(self) -> np.ndarray:
        """calculate the diffusion coefficient matrix of single component"""
        if self._cached_diffusion_ij is None:
            self._cached_diffusion_ij = liquid_para.calculate_diffusion_ij(
                self._temperature, self._tao, self._tao_035,
                self._tao_2, self._tao_3, self._T2, self._T3)
        return self._cached_diffusion_ij

    # 4. single component physical property calculation - external interface
    def density_mole_ij(self, i: Optional[int] = None) -> Union[float, np.ndarray]:
        return get_matrix_result(self._density_mole_ij, i)
        
    def density_mass_ij(self, i: Optional[int] = None) -> Union[float, np.ndarray]:
        return get_matrix_result(self._density_mass_ij, i)
        
    def cp_mole_ij(self, i: Optional[int] = None) -> Union[float, np.ndarray]:
        return get_matrix_result(self._cp_mole_ij, i)
        
    def cp_mass_ij(self, i: Optional[int] = None) -> Union[float, np.ndarray]:
        return get_matrix_result(self._cp_mass_ij, i)
        
    def heat_vaporization_mole_ij(self, i: Optional[int] = None) -> Union[float, np.ndarray]:
        return get_matrix_result(self._heat_vaporization_mole_ij, i)
        
    def heat_vaporization_mass_ij(self, i: Optional[int] = None) -> Union[float, np.ndarray]:
        return get_matrix_result(self._heat_vaporization_mass_ij, i)
        
    def thermal_conductivity_ij(self, i: Optional[int] = None) -> Union[float, np.ndarray]:
        return get_matrix_result(self._thermal_conductivity_ij, i)
        
    def vapor_pressure_ij(self, i: Optional[int] = None) -> Union[float, np.ndarray]:
        return get_matrix_result(self._vapor_pressure_ij, i)
        
    def viscosity_ij(self, i: Optional[int] = None) -> Union[float, np.ndarray]:
        return get_matrix_result(self._viscosity_ij, i)
    
    # 5. mixture physical property calculation
    @property
    def molecular_weight(self) -> float:
        """calculate the average molecular weight of the mixture"""
        if self._cached_molecular_weight is None:
            self._cached_molecular_weight = liquid_para.calculate_molecular_weight_mean(self.composition)
        return self._cached_molecular_weight

    @property
    def density_mole(self) -> float:
        """calculate the mole density of the mixture"""
        if self._cached_density_mole is None:
            self._cached_density_mole = liquid_para.calculate_density_mole_mean(self.composition, self._density_mole_ij)
        return self._cached_density_mole

    @property
    def density_mass(self) -> float:
        """calculate the mass density of the mixture"""
        if self._cached_density_mass is None:
            self._cached_density_mass = self.density_mole * self.molecular_weight
        return self._cached_density_mass

    @property
    def cp_mole(self) -> float:
        """calculate the mole heat capacity of the mixture"""
        if self._cached_cp_mole is None:
            self._cached_cp_mole = liquid_para.calculate_cp_mole_mean(self.composition, self._cp_mole_ij)
        return self._cached_cp_mole

    @property
    def cp_mass(self) -> float:
        """calculate the mass heat capacity of the mixture"""
        if self._cached_cp_mass is None:
            self._cached_cp_mass = self.cp_mole / self.molecular_weight
        return self._cached_cp_mass

    @property
    def heat_vaporization_mole(self) -> float:
        """calculate the mole heat of vaporization of the mixture"""
        if self._cached_heat_vaporization_mole is None:
            self._cached_heat_vaporization_mole = liquid_para.calculate_heat_vaporization_mole_mean(self.composition, self._heat_vaporization_mole_ij)
        return self._cached_heat_vaporization_mole

    @property
    def heat_vaporization_mass(self) -> float:
        """calculate the mass heat of vaporization of the mixture"""
        if self._cached_heat_vaporization_mass is None:
            self._cached_heat_vaporization_mass = self.heat_vaporization_mole / self.molecular_weight
        return self._cached_heat_vaporization_mass

    @property
    def thermal_conductivity(self) -> float:
        """calculate the thermal conductivity of the mixture"""
        if self._cached_thermal_conductivity is None:
            self._cached_thermal_conductivity = liquid_para.calculate_thermal_conductivity_mean(self.composition, self._thermal_conductivity_ij, self.mass_fraction)
        return self._cached_thermal_conductivity

    @property
    def vapor_pressure(self) -> float:
        """calculate the saturated vapor pressure of the mixture"""
        if self._cached_vapor_pressure is None:
            self._cached_vapor_pressure = liquid_para.calculate_vapor_pressure_mean(self.composition, self._vapor_pressure_ij)
        return self._cached_vapor_pressure

    @property
    def viscosity(self) -> float:
        """calculate the viscosity of the mixture"""
        if self._cached_viscosity is None:
            self._cached_viscosity = liquid_para.calculate_viscosity_mean(self.composition, self._viscosity_ij)
        return self._cached_viscosity

    @property
    def reduced_temperature(self) -> np.ndarray:
        return self._Tr

    @property
    def activity_coefficient(self) -> np.ndarray:
        """calculate the activity coefficient"""
        if self._cached_activity_coefficient is None:
            self._cached_activity_coefficient = liquid_para.calculate_activity_coefficients(self._composition, self._temperature)
        return self._cached_activity_coefficient
    

    # auxiliary method
    def get_reduced_temperature(self, i: int) -> float:
        return self._Tr[i]
    
    def activity_coefficient_ij(self, i: Optional[int] = None) -> Union[float, np.ndarray]:
        """calculate the activity coefficient

        Args:
            i: index of the component, if None, return the activity coefficient of all components

        Returns:
            Union[float, np.ndarray]: activity coefficient, array of shape (40,) or single value
        """
        activity_matrix = liquid_para.calculate_activity_coefficients(
            self._composition, 
            self._temperature
        )
        
        if i is not None:
            return activity_matrix[i]
        else:
            return activity_matrix


            
    def diffusion_ij(self, i1: Optional[int] = None, j1: Optional[int] = None, 
                    i2: Optional[int] = None, j2: Optional[int] = None) -> Union[float, np.ndarray]:
        """get the diffusion coefficient of the specified component
        
        Args:
            i1: index of the component 1, optional
            j1: index of the component 1, optional
            i2: index of the component 2, optional
            j2: index of the component 2, optional
            
        Returns:
            Union[float, np.ndarray]: diffusion coefficient
        """
        if i1 is not None and j1 is not None and i2 is not None and j2 is not None:
            return self._diffusion_ij[i1, j1, i2, j2]
        elif i1 is not None and j1 is not None:
            return self._diffusion_ij[i1, j1, :, :]
        elif i2 is not None and j2 is not None:
            return self._diffusion_ij[:, :, i2, j2]
        else:
            return self._diffusion_ij
        
    @property
    def diffusion_mean(self) -> np.ndarray:
        """calculate the average diffusion coefficient
        
        Returns:
            np.ndarray: array of shape (40,), representing the average diffusion coefficient of each component
        """
        if self._cached_diffusion_mean is None:
            self._cached_diffusion_mean = liquid_para.calculate_diffusion_mean(
                self._composition, 
                self._temperature,
                self._Tr,
                self._tao,
                self._tao_035,
                self._tao_2,
                self._tao_3,
                self._T2,
                self._T3
            )
        return self._cached_diffusion_mean

    @property
    def species_names(self) -> list:
        """return the names of all components in the liquid phase (length 40)"""
        return [LIQUID_SPECIES_NAMES[i] for i in range(40)]