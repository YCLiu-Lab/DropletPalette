"""
liquid mass fraction solver module for evaporation

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.
"""

import time
import numpy as np
import numba
import scipy
from scipy.integrate._ivp.ivp import OdeResult
from scipy.sparse import coo_matrix
from scipy.integrate import solve_ivp
from src.solution import LiquidArray
from src.core import Grid, Surface
from .math_utils import ThreePointsScheme, EquationProperty

class LiquidSolverYEvap:
    """liquid phase solver - mass fraction field"""
    
    def __init__(self, grid: Grid, surface: Surface, liquid_00, three_points_scheme: ThreePointsScheme, equation_property: EquationProperty, liquid_array_iter: LiquidArray):
        """initialize the mass fraction field solver
        
        Args:
            grid: grid object
            surface: surface object
            liquid_00: droplet center liquid phase state
            three_points_scheme: three points scheme object
            equation_property: equation property object
            liquid_array_iter: liquid phase iteration array
        """
        self.grid = grid
        self.surface = surface
        self.liquid_00 = liquid_00
        self.three_points_scheme = three_points_scheme
        self.equation_property = equation_property
        self.liquid_array_iter = liquid_array_iter
        
        # get the number of cells and species
        self.n_cells = self.grid.liquid_grid.cell_count
        self.n_species = len(self.liquid_array_iter.liquids[0].composition)
        
        # initialize the mass generation rate
        self.mass_rate_liquid = np.zeros(self.n_species)  # mass generation rate of each species
        
        # initialize the jacobian matrix sparsity structure
        self.jac_sparsity = self._init_jacobian_sparsity()
        
        # initialize the error tolerance
        self.atol = np.full(self.n_cells * self.n_species, 1e-6)  # absolute error tolerance of mass fraction
        self.rtol = 1e-3  # relative error tolerance
        
        # pre-allocate the intermediate arrays
        self._preallocate_arrays()
        
    def _preallocate_arrays(self):
        """pre-allocate the intermediate arrays needed in the calculation"""
        n_cells = self.n_cells
        n_species = self.n_species
        
        # species equation related arrays
        self._species_diffusion_term = np.zeros((n_cells, n_species))
        self._dYdt = np.zeros((n_cells, n_species))  

        # pre-allocate the derivative vector (as a view of _dYdt)
        self._dydt = self._dYdt.reshape(n_cells * n_species, order='F')
        # pre-allocate the mass fraction matrix
        self._Y = np.zeros((n_cells, n_species))

    def _init_jacobian_sparsity(self) -> scipy.sparse.csr_matrix:
        """
        initialize the sparsity structure of the mass fraction jacobian matrix.
        
        - the mass fraction of each grid point is related to itself and the two adjacent grid points
        
        Returns:
            scipy.sparse.csr_matrix: CSR format sparse matrix, representing the position of non-zero elements in the mass fraction jacobian matrix
        """
        # get the number of cells and species
        n_cells = self.n_cells
        n_species = self.n_species
        
        # the total size of the state vector = mass fraction (n_cells * n_species)
        n_total = n_cells * n_species
        
        # define the offsets of the three-point difference (left 1, center, right 1)
        offsets = np.array([-1, 0, 1])
        
        # initialize the row and column index lists
        rows = []
        cols = []
        
        # mass fraction equation (0 to n_total-1)
        for species_idx in range(n_species):
            for i in range(n_cells):
                # calculate the equation index of the current species in the current grid
                eq_idx = i + species_idx * n_cells
                
                # calculate the possible adjacent grid indices
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
        
        # print the statistics of the sparse matrix
        print("\n=== the sparse matrix of the liquid phase mass fraction jacobian matrix ===")
        print(f"    matrix size: {n_total} x {n_total}")
        print(f"    number of non-zero elements: {nnz}")
        print(f"    sparsity: {nnz/(n_total*n_total)*100:.2f}%")
        print("="*50+"\n")
        
        # convert to CSR format and return
        return sparsity.tocsr()
    
    def liquid_governing_equations(self, t: float, Y_flat: np.ndarray) -> np.ndarray:
        """governing equations of liquid phase mass fraction field
        
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
        
        # update the liquid phase state
        self.liquid_array_iter.TPY(self.liquid_array_iter.T, self.liquid_array_iter.P, self._Y)
        self.equation_property.liquid_update_equation_property_Y(self.liquid_array_iter, self.grid, self.liquid_00, self.surface, self.three_points_scheme, self._Y)
        
        # reset the derivative vector
        self._dYdt.fill(0.0)
        
        # calculate the species equation terms
        self._compute_species_terms(
            self._Y,
            self.liquid_array_iter.density_mass,
            self.grid.liquid_grid.volumes,
            self.equation_property.rho_diffusivity_right_center,
            self.equation_property.mass_fraction_gradient_right,
            self.equation_property.rho_diffusivity_left_center,
            self.equation_property.mass_fraction_gradient_left,
            self.grid.liquid_grid.areas_right_boundary,
            self.grid.liquid_grid.areas_left_boundary,
            self._species_diffusion_term,
            self._dYdt,                 
            self.mass_rate_liquid
        )
        self._dydt = self._dYdt.reshape(n_cells * n_species, order='F')                                                     
        return self._dydt.copy()  # return the copy instead of the original array
    
    def solve_bdf(self, Dt: float, y0: np.ndarray, mass_rate_liquid: np.ndarray = None) -> OdeResult:
        """using BDF method to solve the mass fraction governing equation
        
        Args:
            Dt: time step
            y0: initial condition vector, containing mass fraction values
            mass_rate_liquid: liquid phase mass generation rate, shape (n_species,), default is 0
            
        Returns:
            OdeResult: solution result
        """
        # start timing
        start_time = time.time()
        
        # set the mass generation rate
        if mass_rate_liquid is not None:
            self.mass_rate_liquid = mass_rate_liquid.copy()
        else:
            self.mass_rate_liquid.fill(0.0)
        
        # using solve_ivp to solve
        result = solve_ivp(
            fun=lambda t, Y_flat: self.liquid_governing_equations(t, Y_flat),
            t_span=(0, Dt),
            y0=y0,
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
            
            print(f"** liquid phase mass fraction range: {min_sum:.2e} (grid {min_sum_idx}) - {max_sum:.2e} (grid {max_sum_idx}), solving time: {solve_time:.2f}s")
        else:
            print(f"\n** liquid phase mass fraction field solving failed")
        
        return result

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _compute_species_terms(
        Y, density_mass,
        volumes,
        rho_diffusivity_right_center, mass_fraction_gradient_right,
        rho_diffusivity_left_center, mass_fraction_gradient_left,
        liquid_grid_right_areas, liquid_grid_left_areas,
        species_diffusion_term,
        dYdt,
        mass_rate_liquid
    ):
        """calculate the species equation terms
        
        using numba.jit to accelerate the calculation
        """
        n_cells, n_species = Y.shape
                
        # using range to perform parallel calculation
        for i in range(n_cells):
            # species conservation equation coefficient term
            coef = 1.0 / density_mass[i]
            
            # pre-calculate the common factor of the current grid
            cell_vol_i = volumes[i]
            right_area_i = liquid_grid_right_areas[i]
            left_area_i = liquid_grid_left_areas[i]
            
            # diffusion term - vectorized
            species_diffusion_term[i, :] = (
                rho_diffusivity_right_center[i, :] * mass_fraction_gradient_right[i, :] * right_area_i -
                rho_diffusivity_left_center[i, :] * mass_fraction_gradient_left[i, :] * left_area_i
            ) / cell_vol_i
            
            # calculate the mass fraction derivative - vectorized
            dYdt[i, :] = coef * species_diffusion_term[i, :] 
        
        # add the effect of mass generation rate to the last grid point
        dYdt[-1, :] += -mass_rate_liquid / (density_mass[-1] * volumes[-1])