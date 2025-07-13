"""
core module

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.

This module provides the core functions of the droplet evaporation simulation framework, including:
1. time management (runtime)
2. grid system (grid)
3. data management (data_manager)
"""

from .runtime import Runtime
from .grid import Grid, GridParameters, PhaseGrid
from .data_manager import DataManager
from .surface import Surface

__all__ = [
    'Runtime',
    'Grid', 'GridParameters', 'PhaseGrid',
    'DataManager',
    'Surface'
] 