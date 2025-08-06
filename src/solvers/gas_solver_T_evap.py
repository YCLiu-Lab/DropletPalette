"""
gas temperature solver module for evaporation

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.
"""

import time
import cantera as ct
import numpy as np
import numba
import scipy
from scipy.integrate._ivp.ivp import OdeResult
from scipy.sparse import coo_matrix
from scipy.integrate import solve_ivp
from src.core import Grid, Surface
from .math_utils import ThreePointsScheme, EquationProperty

class GasSolverTEvap:
    """gas phase solver - temperature field"""
    
    def __init__(self, grid: Grid, surface: Surface, gas_inf: ct.Solution, three_points_scheme: ThreePointsScheme, equation_property: EquationProperty,gas_array_iter: ct.SolutionArray,relative_velocity_right: np.ndarray,relative_velocity_left: np.ndarray,flag_fast_solver_for_transient: bool):
        """initialize the temperature field solver
        
        Args:
            grid: grid object
            surface: surface object
            gas_inf: infinity gas phase state
        """
        self.grid = grid
        self.surface = surface
        self.gas_inf = gas_inf
        self.three_points_scheme = three_points_scheme
        self.equation_property = equation_property

        self.gas_array_iter = gas_array_iter

        self._relative_velocity_right = relative_velocity_right
        self._relative_velocity_left = relative_velocity_left
        self.flag_fast_solver_for_transient = flag_fast_solver_for_transient
        # get the number of cells
        self.n_cells = self.grid.gas_grid.cell_count

        # initialize the jacobian matrix sparsity structure
        self.jac_sparsity = self._init_jacobian_sparsity()
        
        # initialize the error tolerance
        self.atol = np.full(self.n_cells, 1e-3)  # absolute error tolerance of temperature
        self.rtol = 1e-3  # relative error tolerance
        
        # pre-allocate the intermediate arrays
        self._preallocate_arrays()
        
    def _preallocate_arrays(self):
        """pre-allocate the intermediate arrays needed in the calculation"""
        n_cells = self.n_cells
        
        # energy equation related arrays
        self._energy_convection_term = np.zeros(n_cells)
        self._energy_conduction_term = np.zeros(n_cells)
        self._dTdt = np.zeros(n_cells)  # used to store the temperature derivative of each grid point
        self._dtdt = np.zeros(n_cells)  # used to store the final derivative vector

    def _init_jacobian_sparsity(self) -> scipy.sparse.csr_matrix:
        """
        initialize the sparsity structure of the temperature jacobian matrix.
        the temperature of each grid point is related to itself and the two adjacent grid points.
        
        Returns:
            scipy.sparse.csr_matrix: CSR format sparse matrix, representing the position of non-zero elements in the temperature jacobian matrix
        """
        # get the number of cells
        n_cells = self.n_cells
        
        # define the offsets of the three-point difference (left 1, center, right 1)
        offsets = np.array([-1, 0, 1])
        
        # initialize the row and column index lists
        rows = []
        cols = []
        
        # add temperature dependency
        for i in range(n_cells):
            # calculate the possible adjacent grid indices
            neighbor_indices = i + offsets
            
            # filter out invalid indices
            valid_indices = neighbor_indices[(neighbor_indices >= 0) & (neighbor_indices < n_cells)]
            
            # temperature depends on the temperature of the adjacent grid
            for j in valid_indices:
                rows.append(i)  # temperature equation i
                cols.append(j)  # depends on temperature j
        
        # generate the data array (all 1 boolean array)
        data = np.ones(len(rows), dtype=bool)
        
        # create the COO format sparse matrix
        sparsity = coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells), dtype=bool)
        nnz = sparsity.getnnz()
        
        # print the statistics of the sparse matrix
        print("\n=== the sparse matrix of the gas phase temperature jacobian matrix ===")
        print(f"    matrix size: {n_cells} x {n_cells}")
        print(f"    number of non-zero elements: {nnz}")
        print(f"    sparsity: {nnz/(n_cells*n_cells)*100:.2f}%")
        print("="*50+"\n")
        
        # convert to CSR format and return
        return sparsity.tocsr()
        
   
    def gas_governing_equations(self, t: float, T: np.ndarray) -> np.ndarray:
        """temperature field governing equation right side
        
        Args:
            t: time
            T: temperature vector, shape (n_cells,)
            
        Returns:
            np.ndarray: temperature derivative vector, shape (n_cells,)
        """
        
        # update the gas phase state
        self.gas_array_iter.TPY = T, self.gas_array_iter.P, self.gas_array_iter.Y
        self.equation_property.gas_update_equation_property_T(self.gas_array_iter, self.grid, self.surface, self.gas_inf, self.three_points_scheme)
        
        # reset the derivative vector
        self._dTdt.fill(0.0)
        self._dtdt.fill(0.0)
        
        # calculate the energy equation terms
        self._compute_energy_terms(
            self.gas_array_iter.density_mass,
            self.gas_array_iter.cp_mass,
            self.equation_property.specific_enthalpy_right_upwind,
            self.equation_property.temperature_right_upwind,
            self._relative_velocity_right,
            self.grid.gas_grid.areas_right_boundary,
            self.equation_property.specific_enthalpy_left_upwind,
            self.equation_property.temperature_left_upwind,
            self._relative_velocity_left,
            self.grid.gas_grid.areas_left_boundary,
            self.grid.gas_grid.volumes,
            self.equation_property.thermal_conductivity_right_center,
            self.equation_property.temperature_gradient_right,
            self.equation_property.thermal_conductivity_left_center,
            self.equation_property.temperature_gradient_left,
            self._energy_convection_term,
            self._energy_conduction_term,
            self._dTdt
        )
        
        # write the temperature derivative into the result vector
        self._dtdt[:] = self._dTdt[:]
        
        return self._dtdt.copy()  # return the copy instead of the array
    
    def solve_bdf(self, Dt: float, T0: np.ndarray) -> OdeResult:
        """using BDF method to solve the temperature governing equation
        Args:
            Dt: time step
            T0: initial temperature vector, shape (n_cells,)
            
        Returns:
            OdeResult: solution result
        """
        # start timing
        start_time = time.time()
        
        self.gas_array_iter.TPY = T0, self.gas_array_iter.P, self.gas_array_iter.Y
        # using solve_ivp to solve
        result = solve_ivp(
            fun=lambda t, T: self.gas_governing_equations(t, T),
            t_span=(0, Dt),
            y0=T0,
            method='BDF',
            rtol=self.rtol,
            atol=self.atol,
            jac_sparsity=self.jac_sparsity
        )
        
        # calculate the solving time
        solve_time = time.time() - start_time
        
        # output the solving information
        if result.success:
            # calculate the maximum and minimum temperature
            max_temp = np.max(result.y[:,-1])
            min_temp = np.min(result.y[:,-1])
            max_temp_idx = np.argmax(result.y[:,-1])
            min_temp_idx = np.argmin(result.y[:,-1])
            
            print(f"** gas phase temperature range: {min_temp:.2f}K (grid {min_temp_idx}) - {max_temp:.2f}K (grid {max_temp_idx}), solving time: {solve_time:.2f}s")
        else:
            print(f"\n** temperature field solving failed")
        
        return result

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _compute_energy_terms(
        density_mass, cp_mass, 
        specific_enthalpy_right_upwind,
        temperature_right_upwind, relative_velocity_right, areas_right_boundary,
        specific_enthalpy_left_upwind, temperature_left_upwind,
        relative_velocity_left, areas_left_boundary, volumes,
        thermal_conductivity_right_center, temperature_gradient_right,
        thermal_conductivity_left_center, temperature_gradient_left,
        energy_convection_term, energy_conduction_term,
        dTdt
    ):
        """calculate the energy equation terms - vectorized + JIT version
        
        use numba.jit to accelerate the calculation, and use vectorized operation
        """
        # convection term - vectorized calculation
        energy_convection_term[:] = -(
            specific_enthalpy_right_upwind * temperature_right_upwind * relative_velocity_right * areas_right_boundary -
            specific_enthalpy_left_upwind * temperature_left_upwind * relative_velocity_left * areas_left_boundary
        ) / (volumes * density_mass * cp_mass)
        
        # conduction term - vectorized calculation
        energy_conduction_term[:] = (
            thermal_conductivity_right_center * temperature_gradient_right * areas_right_boundary -
            thermal_conductivity_left_center * temperature_gradient_left * areas_left_boundary
        ) / (volumes * density_mass * cp_mass)
        
        
        # temperature derivative - vectorized calculation
        dTdt[:] = energy_conduction_term + energy_convection_term
