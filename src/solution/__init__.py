"""
solution module

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.

This module provides the solution related functions, including:
1. mapping_utils: mapping tool for species
2. liquid_para: liquid phase parameter definition
3. liquid_utils: liquid phase calculation tool
4. liquid: liquid phase state class
5. liquid_array: liquid phase array class
"""
from .mapping_utils import init_species_mapping
from .liquid import Liquid
from .liquid_array import LiquidArray

__all__ = ['init_species_mapping', 'Liquid', 'LiquidArray']
