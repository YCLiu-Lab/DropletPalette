"""
liquid utils module

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.

This module provides various utility functions for liquid mixtures (liquid, liquid_array) calculations, including:

1. liquid setting validation functions:
   - validate_composition: validate the validity of a single component
   - validate_pressure: validate the validity of pressure
   - validate_temperature: validate the validity of temperature

2. liquid_array setting validation functions:
   - validate_composition_array: validate the validity of composition array
   - validate_composition_array_with_message: validate the validity of composition array with detailed error message
   - validate_pressure_array: validate the validity of pressure array
   - validate_temperature_array: validate the validity of temperature array

3. supercritical species checking functions:
   - find_supercritical_species: check and report supercritical species

4. utility functions:
   - normalize: normalize the composition array
   - get_matrix_result: get the matrix result according to the index
"""

import numpy as np
import numba
from typing import Union, Optional
from .liquid_para import TC, COM_TOLERANCE, SPECIES_NAMES


# constant definition
PRESSURE_MIN = 0.0
TEMPERATURE_MIN = 0.0
NUM_SPECIES = 40    # number of components
# 1. liquid setting validation functions:

@numba.njit(cache=True)
def _validate_composition_impl(composition: np.ndarray) -> None:
    """validate the implementation function of the composition"""
    if composition.size != NUM_SPECIES:
        raise ValueError("Invalid composition size")
    
    if np.any(composition < 0):
        for i in range(composition.size):
            if composition[i] < 0:
                raise ValueError("Negative composition value")
    

@numba.njit(cache=True)
def _validate_pressure_impl(pressure: float) -> None:
    """validate the implementation function of the pressure"""
    if pressure <= PRESSURE_MIN:
        raise ValueError("Invalid pressure")

@numba.njit(cache=True)
def _validate_temperature_impl(temperature: float) -> None:
    """validate the implementation function of the temperature"""
    if temperature <= TEMPERATURE_MIN:
        raise ValueError("Invalid temperature")

# warmup numba function
def _warmup_numba():
    """warmup numba function to avoid the compilation overhead of the first call"""
    try:
        test_comp = np.ones(NUM_SPECIES) / NUM_SPECIES
        _validate_composition_impl(test_comp)
        _validate_pressure_impl(101325.0)
        _validate_temperature_impl(300.0)
    except:
        pass

# warmup numba function when the module is imported
_warmup_numba()

def validate_composition(composition: np.ndarray) -> None:
    """validate the validity of the composition"""
    try:
        _validate_composition_impl(composition)
    except ValueError as e:
        if str(e) == "Invalid composition size":
            raise ValueError(f"composition size error: expected {NUM_SPECIES}, actual {composition.size}")
        elif str(e) == "Negative composition value":
            for i, x in enumerate(composition):
                if x < 0:
                    if x < -1e-3:
                        raise ValueError(f"negative mole fraction: index {i}, value {x}")
                    else:
                        x = 0


def validate_pressure(pressure: float) -> None:
    """validate the validity of the pressure"""
    try:
        _validate_pressure_impl(pressure)
    except ValueError:
        raise ValueError(f"pressure must be greater than {PRESSURE_MIN}Pa, current value: {pressure}Pa")

def validate_temperature(temperature: float) -> None:
    """validate the validity of the temperature"""
    try:
        _validate_temperature_impl(temperature)
    except ValueError:
        raise ValueError(f"temperature must be greater than {TEMPERATURE_MIN}K, current value: {temperature}K")



# 2. liquid_array validation functions:

@numba.njit(cache=True)
def validate_temperature_array(temperature: np.ndarray) -> None:
    """validate the validity of the temperature array, if invalid, raise an exception
    
    Args:
        temperature (np.ndarray): temperature array
        
    Exception:
        ValueError: when the temperature array has invalid values, raise an exception
    """
    if np.any(temperature <= 0):
        raise ValueError("temperature must be greater than 0, invalid values exist")

@numba.njit(cache=True)
def validate_pressure_array(pressure: np.ndarray) -> None:
    """validate the validity of the pressure array, if invalid, raise an exception
    
    Args:
        pressure (np.ndarray): pressure array
        
    Exception:
        ValueError: when the pressure array has invalid values, raise an exception
    """
    if np.any(pressure <= 0):
        raise ValueError("pressure must be greater than 0, invalid values exist")

@numba.njit(cache=True)
def validate_composition_array(composition: np.ndarray) -> None:
    """validate the validity of the composition array, if invalid, raise an exception
    
    Args:
        composition (np.ndarray): composition array, shape (n, 40)
        
    Exception:
        ValueError: when the composition array has invalid values, raise an exception
    """
    # validate the array dimension
    if composition.ndim != 2:
        raise ValueError("Invalid array dimension")
    if composition.shape[1] != NUM_SPECIES:
        raise ValueError("Invalid array shape")
        
    # validate the non-negativity
    if np.any(composition < 0):
        raise ValueError("negative composition values")
        
    # validate the sum of composition of each row is close to 1
    for i in range(composition.shape[0]):
        sum_comp = np.sum(composition[i])
        if abs(sum_comp - 1.0) > COM_TOLERANCE:
            raise ValueError("Sum of composition not 1")

def validate_composition_array_with_message(composition: np.ndarray) -> None:
    """validate the composition array with detailed error message"""
    try:
        validate_composition_array(composition)
    except ValueError as e:
        if str(e) == "Invalid array dimension":
            raise ValueError(f"composition array must be a 2D array, current dimension: {composition.ndim}")
        elif str(e) == "Invalid array shape":
            raise ValueError(f"composition array second dimension must be {NUM_SPECIES}, current shape: {composition.shape}")
        elif str(e) == "Negative composition values":
            raise ValueError("composition must be greater than or equal to 0, invalid values exist")
        elif str(e) == "Sum of composition not 1":
            for i in range(composition.shape[0]):
                sum_comp = np.sum(composition[i])
                if abs(sum_comp - 1.0) > COM_TOLERANCE:
                    raise ValueError(f"sum of composition of row {i} is not 1: {sum_comp}, error {abs(sum_comp-1.0)} > {COM_TOLERANCE}")
        else:
            raise


# 3. supercritical species checking functions:

@numba.njit(cache=True)
def _find_supercritical_species_impl(composition: np.ndarray, temperature: float, tc: np.ndarray) -> None:
    """find the supercritical species"""
    # use vectorized operation to find all supercritical species
    mask = (composition > 0) & (temperature > tc)
    if np.any(mask):
        # only traverse when there are supercritical species to get specific information
        for i in range(len(composition)):
            if mask[i]:
                print("Warning: Supercritical species detected")

def find_supercritical_species(composition: np.ndarray, temperature: float) -> None:
    """find the supercritical species, if exist, raise a warning"""
    _find_supercritical_species_impl(composition, temperature, TC)
    # if there are supercritical species, generate detailed warning information
    mask = (composition > 0) & (temperature > TC)
    if np.any(mask):
        for i in range(len(composition)):
            if mask[i]:
                print(f"Warning: species {SPECIES_NAMES[i]} is supercritical (T = {temperature:.2f}K > Tc = {TC[i]:.2f}K)")

# 4. utility functions:

@numba.njit(cache=True)
def normalize(composition: np.ndarray) -> np.ndarray:
    """normalize the composition array
    
    Args:
        composition: composition array, can be mole fraction or mass fraction
        
    Returns:
        np.ndarray: normalized composition array

    """
    total = np.sum(composition)
    if total > 0:
        return composition / total
    else:
        return np.zeros_like(composition)

# utility function-return the corresponding part of the matrix according to the index
def get_matrix_result(matrix: np.ndarray, 
                     i: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    return the corresponding part of the matrix according to the index.
    if the index is None, return the whole matrix.
    
    Args:
        matrix (np.ndarray): input matrix, shape (40,)
        i (Optional[int]): index value, if None, return the whole matrix
        
    Returns:
        Union[float, np.ndarray]: 
            - if i is None, return the whole matrix
            - if i is a valid index, return the element at the corresponding position
            
    Exception:
        IndexError: when the index is out of the matrix range
    """
    if i is None:
        return matrix
        
    if i < 0 or i >= len(matrix):
        raise IndexError("index %d out of matrix range [0, %d]" % (i, len(matrix)-1))
        
    return matrix[i]