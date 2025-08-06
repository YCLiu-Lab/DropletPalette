"""
gas quasi-steady solver module

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.
"""


"""
quasi-steady gas solver module

This module implements the quasi-steady gas solver, used to calculate the heat and mass transfer during droplet evaporation.
"""

import numpy as np
import cantera as ct
from dataclasses import dataclass
from scipy.optimize import fsolve
from src.core import Grid, Surface

@dataclass
class QSParameters:
    """quasi-steady droplet calculation parameters
    
    Attributes:
        B_M: mass transfer number
        B_T: heat transfer number
        E_i: evaporation rate of each component
        Le: Lewis number
        Q_dot: heat transfer rate
        mdot: mass flux
    """
    y_fs: float = 0.0
    B_M: float = 0.0
    B_T: float = 0.0
    E_i: np.ndarray = None
    Le: float = 0.0
    Q_dot: float = 0.0
    Q_gas: float = 0.0
    Q_hv: float = 0.0
    mdot: float = 0.0
    D_ref: float = 0.0
    dys_dr: float = 0.0
    rho_ref: float = 0.0
class GasSolverQS:
    """quasi-steady gas solver
    
    This solver is based on the quasi-steady assumption, using the film theory to calculate the heat and mass transfer during droplet evaporation.
    """
    def __init__(self, grid: Grid, surface: Surface, gas_inf: ct.Solution, gas_ref: ct.Solution, gas_flux: ct.Solution, liquid_flux,gas_velocity):
        """initialize the quasi-steady gas solver
        
        Args:
            grid: grid object
            surface: surface object
            gas_inf: gas phase infinite far state
        """
        self.grid = grid
        self.gas_array_iter = ct.SolutionArray(grid.gas_array._phase, shape=grid.gas_array.shape)
        self.gas_array_iter.TPX = grid.gas_array.TPX
        self.gas_surface = surface.gas_surface
        self.liquid_surface = surface.liquid_surface
        self.gas_inf = gas_inf
        self.gas_ref = gas_ref
        self.gas_flux = gas_flux
        self.liquid_flux = liquid_flux
        self.radius = grid.params.droplet_radius
        # get the fuel component index
        self.gas_fuel_indices =surface.gas_fuel_indices
        self.liquid_fuel_indices = surface.liquid_fuel_indices
        self.gas_velocity = gas_velocity
        # initialize the calculation parameters
        self.calc_params = None
        self.update_calc_params()
    def update_calc_params(self):
        self._calculate_evap_state()
        self._calculate_reference_state()
        self._calculate_calc_params()
        return self.calc_params

    def _calculate_evap_state(self):
        evap_mole_fractions_gas = np.zeros_like(self.gas_surface.X)
        evap_mole_fractions_gas[self.gas_fuel_indices] = self.gas_surface.X[self.gas_fuel_indices]
        evap_mole_fractions_gas = evap_mole_fractions_gas / np.sum(evap_mole_fractions_gas)
        evap_mole_fractions_liquid = np.zeros_like(self.liquid_surface.composition)
        evap_mole_fractions_liquid[self.liquid_fuel_indices] = evap_mole_fractions_gas[self.gas_fuel_indices]
        self.gas_flux.TPX = self.gas_surface.T, self.gas_surface.P, evap_mole_fractions_gas
        self.liquid_flux.TPX = self.gas_surface.T, self.gas_surface.P, evap_mole_fractions_liquid
    def _calculate_reference_state(self):
        """calculate the reference state"""
        # calculate the reference temperature
        T_ref = (2/3) * self.gas_surface.T + (1/3) * self.gas_inf.T
        # calculate the reference composition
        X_ref = (2/3) * self.gas_surface.X + (1/3) * self.gas_inf.X
        # update the reference state
        self.gas_ref.TPX = T_ref, self.gas_ref.P, X_ref
        self.gas_flux.TPX = T_ref, self.gas_flux.P, self.gas_flux.X
    def myfun_B_T(self,B_T,B_M, Sh_mod,Nu_0,Le):
        Nu_mod = 2+(Nu_0-2)*B_T/(1+B_T)**0.7/np.log(1+B_T)
        F = B_T+1-(1+B_M)**((Sh_mod/Le)/(Nu_mod)*(self.gas_flux.cp_mass/self.gas_ref.cp_mass))
        return F
    def _calculate_calc_params(self):
        # calculate the Spalding mass transfer number
        D_ref = self.gas_ref.mix_diff_coeffs_mass[self.gas_fuel_indices]@self.gas_flux.Y[self.gas_fuel_indices]
        Y_s = self.gas_surface.Y[self.gas_fuel_indices]
        Y_inf = self.gas_inf.Y[self.gas_fuel_indices]
        Le = self.gas_ref.thermal_conductivity/ (self.gas_ref.cp_mass * self.gas_ref.density_mass * D_ref)
        B_M = (np.sum(Y_s) - np.sum(Y_inf)) / (1 - np.sum(Y_s))
        Re = self.gas_ref.density_mass * self.gas_velocity * 2*self.grid.params.droplet_radius / self.gas_ref.viscosity
        Pr = self.gas_ref.cp_mass * self.gas_ref.viscosity / self.gas_ref.thermal_conductivity
        Sc = self.gas_ref.viscosity / (self.gas_ref.density_mass * D_ref)
        if Re < 1:
            Nu_0 = 1+(1+Re*Pr)**(1/3)
            Sh_0 = 1+(1+Re*Sc)**(1/3)
        elif Re >=1 and Re <=400:
            Nu_0 = 1+(1+Re*Pr)**(1/3)*Re**0.077
            Sh_0 = 1+(1+Re*Sc)**(1/3)*Re**0.077
        else:
            Nu_0 = 2+0.552*Re**0.5*Pr**(1/3)
            Sh_0 = 2+0.552*Re**0.5*Sc**(1/3)
        Sh_mod = 2+(Sh_0-2)/((1+B_M)**0.7*np.log(1+B_M)/B_M)
        # ignore 4*pi, to keep consistent with other places in the code
        mdot =1/2 * self.grid.params.droplet_radius * self.gas_ref.density_mass * D_ref * np.log(1 + B_M) * Sh_mod
        B_T0 = np.exp(np.log(1 + B_M)*(self.gas_flux.cp_mass/self.gas_ref.cp_mass) / Le) - 1

        B_T = float(fsolve(self.myfun_B_T,B_T0,args=(B_M, Sh_mod,Nu_0,Le))[0])
        
        L_eff = self.gas_flux.cp_mass * (self.gas_inf.T - self.gas_surface.T) / B_T
        L_Q = L_eff - self.liquid_flux.heat_vaporization_mass
        Q_dot = mdot * L_Q
        dys_dr = -((np.sum(Y_s) - np.sum(Y_inf)) / self.grid.params.droplet_radius) * np.log(1 + B_M)/B_M
        Q_gas = mdot * L_eff
        Q_hv = mdot * self.liquid_flux.heat_vaporization_mass

        # calculate the evaporation rate of each component
        E_i = (Y_s - Y_inf) / B_M + Y_s
        self.calc_params = QSParameters(y_fs=np.sum(Y_s),B_M = B_M, B_T = B_T, E_i = E_i, Le = Le, Q_dot = Q_dot, Q_gas = Q_gas,Q_hv = Q_hv,mdot = mdot,D_ref = D_ref,rho_ref = self.gas_ref.density_mass,dys_dr = dys_dr)
    
