"""
DCM_droplet_1D simulation framework

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.

This module provides the main functions of the droplet evaporation simulation, including:
1. liquid phase calculation module (solution)
2. core function module (core)
3. solver module (solvers)
"""

from . import solution
from . import core
from . import solvers
from .simulation import Simulation, SimulationParameters

__all__ = ['solution', 'core', 'solvers', 'Simulation', 'SimulationParameters'] 