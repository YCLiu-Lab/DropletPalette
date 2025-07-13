"""
solver module

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.
"""

"""
solver module for droplet combustion simulation

This module provides the solver functions for the simulation framework, including:
1. gas solver (gas_solver, gas_solver_T_evap, gas_solver_Y_evap, gas_solver_QS)
2. liquid solver (liquid_solver, liquid_solver_T_evap, liquid_solver_Y_evap)

"""
from .gas_solver import GasSolver
from .gas_solver_T_evap import GasSolverTEvap
from .gas_solver_Y_evap import GasSolverYEvap
from .gas_solver_QS import GasSolverQS
from .liquid_solver import LiquidSolver
from .liquid_solver_T_evap import LiquidSolverTEvap
from .liquid_solver_Y_evap import LiquidSolverYEvap

__all__ = [
    'GasSolver',
    'GasSolverTEvap',
    'GasSolverYEvap',
    'GasSolverQS',
    'LiquidSolver',
    'LiquidSolverTEvap',
    'LiquidSolverYEvap'
]
