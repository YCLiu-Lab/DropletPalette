"""
gas mass fraction solver module for evaporation

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
class GasSolverYEvap:
    """gas phase solver - mass fraction field"""
    
    def __init__(self, grid: Grid, surface: Surface, gas_inf: ct.Solution, three_points_scheme: ThreePointsScheme, equation_property: EquationProperty,gas_array_iter: ct.SolutionArray,relative_velocity_right: np.ndarray,relative_velocity_left: np.ndarray,flag_fast_solver_for_transient: bool):
        """initialize the mass fraction field solver
        
        Args:
            grid: grid object
            surface: surface object
            gas_inf: infinity gas phase state
            three_points_scheme: three points scheme object
            equation_property: equation property object
        """
        self.grid = grid
        self.surface = surface
        self.gas_inf = gas_inf
        self.three_points_scheme = three_points_scheme
        self.equation_property = equation_property
        self.gas_array_iter = gas_array_iter
        self.flag_fast_solver_for_transient = flag_fast_solver_for_transient
        # get the number of cells and species
        self.n_cells = self.grid.gas_grid.cell_count
        self.n_species_total = self.gas_array_iter.n_species
        self.n_species = self.n_species_total - 1 # exclude the first specie N2
        
        
        # initialize the jacobian matrix sparsity structure
        self.jac_sparsity = self._init_jacobian_sparsity()
        self._relative_velocity_right = relative_velocity_right
        self._relative_velocity_left = relative_velocity_left

        # initialize the error tolerance
        self.atol = np.full(self.n_cells * self.n_species, 1e-6)  # absolute error tolerance of mass fraction
        self.rtol = 1e-3  # relative error tolerance
        
        # pre-allocate the intermediate arrays
        self._preallocate_arrays()
        
    def _preallocate_arrays(self):
        """pre-allocate the intermediate arrays"""
        n_cells = self.n_cells
        n_species = self.n_species
        n_species_total = self.n_species_total
        # species equation related arrays
        self._species_convection_term = np.zeros((n_cells, n_species))
        self._species_diffusion_term = np.zeros((n_cells, n_species))
        self._dYdt = np.zeros((n_cells, n_species))
        
        # pre-allocate the derivative vector (as a view of _dYdt)
        self._dydt = self._dYdt.reshape(n_cells * n_species, order='F')
        
        # pre-allocate the mass fraction matrix
        self._Y = np.zeros((n_cells, n_species))
        self._Y_total = np.zeros((n_cells, n_species_total))
    
    def _init_jacobian_sparsity(self) -> scipy.sparse.csr_matrix:
        """
        initialize the sparsity structure of the mass fraction jacobian matrix
        
        - the mass fraction of each grid point is related to its temperature and the mass fraction of the related species in the reaction,
          and also related to the mass fraction of the same species in the adjacent two grid points
        
        Returns:
            scipy.sparse.csr_matrix: CSR format sparse matrix, representing the position of the non-zero elements in the mass fraction jacobian matrix
        """
        # get the number of cells and species
        n_cells = self.n_cells
        n_species = self.n_species
        
        # the total size of the state vector = mass fraction (n_cells * n_species)
        n_total = n_cells * n_species
        
        # define the offset of the three-point difference (left 1, center, right 1)
        offsets = np.array([-1, 0, 1])
        
        # initialize the row and column index lists
        rows = []
        cols = []
        
        
        # mass fraction equation (0 to n_total-1)
        for species_idx in range(n_species):
            for i in range(n_cells):
                # calculate the equation index of the current species in the current grid
                eq_idx = i + species_idx * n_cells
                
                # calculate the possible neighbor grid indices
                neighbor_indices = i + offsets
                
                # filter out invalid indices
                valid_indices = neighbor_indices[(neighbor_indices >= 0) & (neighbor_indices < n_cells)]
                
                # mass fraction depends on the same species mass fraction of the adjacent grid points
                for j in valid_indices:
                    rows.append(eq_idx)   # mass fraction equation
                    cols.append(j + species_idx * n_cells)  # depends on mass fraction Y_species,j
        
        # generate the data array (all 1 boolean array)
        data = np.ones(len(rows), dtype=bool)
        
        # create the COO format sparse matrix
        sparsity = coo_matrix((data, (rows, cols)), shape=(n_total, n_total), dtype=bool)
        nnz = sparsity.getnnz()
        print("\n=== the sparse matrix of the gas phase mass fraction jacobian matrix ===")
        print(f"    matrix size: {n_cells*n_species} x {n_cells*n_species}")
        print(f"    number of non-zero elements: {nnz}")
        print(f"    sparsity: {nnz/(n_cells*n_species*n_cells*n_species)*100:.2f}%")
        print("="*50+"\n")
        
        # convert to CSR format and return
        return sparsity.tocsr()
    

    def gas_governing_equations(self, t: float, Y_flat: np.ndarray) -> np.ndarray:
        """governing equations of gas phase mass fraction field
        
        Args:
            t: time
            Y_flat: mass fraction vector, shape (n_cells*n_species,)
            
        Returns:
            np.ndarray: mass fraction derivative vector, shape (n_cells*n_species,)
        """
        n_cells = self.n_cells
        n_species = self.n_species
        # directly use reshape to create a view
        self._Y = Y_flat.reshape(n_cells, n_species, order='F')
        
        # build _Y_total by adding N2 concentration at the front
        # N2 concentration is calculated as 1 - sum of other species
        n2_concentration = 1.0 - np.sum(self._Y, axis=1, keepdims=True)
        self._Y_total = np.concatenate([n2_concentration, self._Y], axis=1)
        
        # update the gas phase state
        self.gas_array_iter.TPY = self.gas_array_iter.T, self.gas_array_iter.P, self._Y_total
        
        self.calculate_gas_relative_velocity_during_Y(self._relative_velocity_left[0])

        if self.flag_fast_solver_for_transient == False:
            self.equation_property.gas_update_equation_property_Y(self.gas_array_iter, self.grid, self.surface, self.gas_inf, self.three_points_scheme, self._Y_total)
        else:
            self.equation_property.gas_update_equation_property_Y_fast(self.gas_array_iter, self.grid, self.surface, self.gas_inf, self.three_points_scheme, self._Y_total)
        
        # reset the derivative vector
        self._dYdt.fill(0.0)
        
        # calculate the species equation terms
        self._compute_species_terms(
            self._Y,
            self.gas_array_iter.density_mass,
            self.equation_property.density_right_upwind,
            self._relative_velocity_right,
            self.grid.gas_grid.areas_right_boundary,
            self.grid.gas_grid.volumes,
            self.equation_property.density_left_upwind,
            self._relative_velocity_left,
            self.grid.gas_grid.areas_left_boundary,
            self.equation_property.mass_fraction_right_upwind[:, 1:],
            self.equation_property.mass_fraction_left_upwind[:, 1:],
            self.equation_property.rho_diffusivity_right_center[:, 1:],
            self.equation_property.mass_fraction_gradient_right[:, 1:],
            self.equation_property.rho_diffusivity_left_center[:, 1:],
            self.equation_property.mass_fraction_gradient_left[:, 1:],
            self._species_convection_term,
            self._species_diffusion_term,
            self._dYdt
        )
        self._dydt = self._dYdt.reshape(n_cells * n_species, order='F')
        return self._dydt.copy()
    
    def solve_bdf(self, Dt: float, y0: np.ndarray) -> OdeResult:
        """using BDF method to solve the mass fraction governing equation
        
        Args:
            Dt: time step
            y0: initial condition vector, containing mass fraction values
            
        Returns:
            OdeResult: solution result
        """
        # start timing
        start_time = time.time()
        n2_concentration = 1.0 - np.sum(self._Y, axis=1, keepdims=True)
        self._Y_total = np.concatenate([n2_concentration, self._Y], axis=1)
        if self.flag_fast_solver_for_transient == True:
            self.equation_property.gas_update_equation_property_Y(self.gas_array_iter, self.grid, self.surface, self.gas_inf, self.three_points_scheme, self._Y_total)
        # using solve_ivp to solve
        result = solve_ivp(
            fun=lambda t, Y_flat: self.gas_governing_equations(t, Y_flat),
            t_span=(0, Dt),
            y0=y0,
            max_step=Dt/100,
            method='BDF',
            rtol=self.rtol,
            atol=self.atol,
            jac_sparsity=self.jac_sparsity
        )

        # calculate the solving time
        solve_time = time.time() - start_time
        
        # output the solving information
        if result.success:
            # calculate the maximum and minimum of the sum of mass fractions
            n_cells = self.n_cells
            n_species = self.n_species
            
            # calculate the sum of mass fractions of each grid
            mass_fraction_sums = np.zeros(n_cells)
            final_Y = result.y[:, -1]  # get the data of the last time step
            
            # add the mass fractions of all species at each grid point
            for i in range(n_cells):
                for species_idx in range(n_species):
                    idx = species_idx * n_cells + i  # calculate the correct index in final_Y
                    mass_fraction_sums[i] += final_Y[idx]
            
            # calculate the maximum and minimum of the sum of mass fractions
            max_sum = np.max(mass_fraction_sums)
            min_sum = np.min(mass_fraction_sums)
            max_sum_idx = np.argmax(mass_fraction_sums)
            min_sum_idx = np.argmin(mass_fraction_sums)
            
            print(f"** gas phase mass fraction range: {min_sum:.2f} (grid {min_sum_idx}) - {max_sum:.2f} (grid {max_sum_idx}), solving time: {solve_time:.2f}s")
        else:
            print(f"** gas phase mass fraction field solving failed")
        
        return result

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _compute_species_terms(
        Y, density_mass,
        density_right_upwind,
        relative_velocity_right, areas_right_boundary, volumes,
        density_left_upwind, relative_velocity_left, areas_left_boundary,
        mass_fraction_right_upwind, mass_fraction_left_upwind,
        rho_diffusivity_right_center, mass_fraction_gradient_right,
        rho_diffusivity_left_center, mass_fraction_gradient_left,
        species_convection_term, species_diffusion_term,
        dYdt
    ):
        """calculate the species equation terms
        
        using numba.jit to accelerate the calculation
        """
        n_cells, n_species = Y.shape
        
        # pre-calculate the common terms
        conv_right_factor = -density_right_upwind * relative_velocity_right * areas_right_boundary / volumes
        conv_left_factor = density_left_upwind * relative_velocity_left * areas_left_boundary / volumes
        
        for i in range(n_cells):
            # species conservation equation coefficient term
            coef = 1.0 / density_mass[i]
            
            # pre-calculate the common terms of the current grid
            conv_right_i = conv_right_factor[i]
            conv_left_i = conv_left_factor[i]
            cell_vol_i = volumes[i]
            right_area_i = areas_right_boundary[i]
            left_area_i = areas_left_boundary[i]
            
            # convection term - vectorized
            species_convection_term[i, :] = coef*(
                conv_right_i * mass_fraction_right_upwind[i, :] +
                conv_left_i * mass_fraction_left_upwind[i, :]
            )
            
            # diffusion term - vectorized
            species_diffusion_term[i, :] = coef*(
                rho_diffusivity_right_center[i, :] * mass_fraction_gradient_right[i, :] * right_area_i -
                rho_diffusivity_left_center[i, :] * mass_fraction_gradient_left[i, :] * left_area_i
            ) / cell_vol_i
            
            # calculate the mass fraction derivative - vectorized
            dYdt[i, :] = species_convection_term[i, :] +species_diffusion_term[i, :]

    def calculate_gas_relative_velocity_during_Y(self, gas_stefan_velocity):
        """calculate the gas phase relative velocity"""
        left_areas = self.grid.gas_grid.areas_left_boundary
        left_density = np.concatenate([[self.surface.gas_surface.density_mass], self.gas_array_iter.density_mass[:-1]])
        right_areas = self.grid.gas_grid.areas_right_boundary
        right_density = self.gas_array_iter.density_mass
        rel_vel = np.zeros(self.grid.gas_grid.cell_count+1)
        # set the first grid
        rel_vel[0] = gas_stefan_velocity
        self._relative_velocity_left[0] = gas_stefan_velocity
        # calculate the recursive factor
        factors = left_areas * left_density / (right_areas * right_density)
        # multiply to get the relative velocity of all grids
        rel_vel[1:] = rel_vel[0] * np.cumprod(factors)
        
        self._relative_velocity_right[:] = rel_vel[1:]
        self._relative_velocity_left[:] = rel_vel[:-1]