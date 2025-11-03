"""
droplet evaporation simulation module

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.
"""

"""
Main classes:
- SimulationParameters: simulation setting parameters class
- Simulation: main simulation class
- DropletParameters: droplet holistic parameters class
- TemperatureChangeTracker: combustion reaction temperature change tracker class
"""

import numpy as np
import cantera as ct
import warnings
from typing import Optional
from dataclasses import dataclass
from src.solution import Liquid, LiquidArray
from src.core import Surface, Runtime, Grid, GridParameters, DataManager
from src.solvers import GasSolver, LiquidSolver, GasSolverQS
# ignore thermodynamic warning of components
warnings.filterwarnings("ignore", category=UserWarning, message=".*NasaPoly2.*")


@dataclass
class DropletParameters:
    """droplet holistic parameters class
    
    Attributes:
        mass_0: initial mass [kg]
        volume_0: initial volume [m³]
        mass_i_0: initial component mass [kg]
        mass: current mass [kg]
        volume: current volume [m³]
        mass_new: new mass [kg]
        volume_new: new volume [m³]
        mass_i: current component mass [kg]
        mass_i_new: new component mass [kg]
    """
    mass_0: float = 0.0
    volume_0: float = 0.0
    mass_i_0: np.ndarray = None
    mass: float = 0.0
    volume: float = 0.0
    mass_i: np.ndarray = None
    mass_new: float = 0.0
    volume_new: float = 0.0
    mass_i_new: np.ndarray = None

@dataclass
class SimulationParameters:
    """simulation setting parameters class
    
    Attributes:
        case_name: case name
        fuel_composition: fuel composition
        droplet_radius: initial droplet radius [m]
        boundary_temperature: boundary temperature [K]
        initial_pressure: initial pressure [bar]
        initial_temperature: initial temperature [K]
        liquid_cell_count: liquid phase grid count
        gas_cell_count: gas phase grid count
        initital_time_step: initial time step [s]
        mechanism_file: reaction mechanism file
        liquid_solver_type: liquid phase solver type ('ITCID'/'FTCID'/'FTCFD'/'ITCFD')
        gas_solver_type: gas phase solver type ('react'/'evap'/'Quasi_Steady')
        gas_velocity: gas phase velocity [m/s], only used when gas_solver_type is 'Quasi_Steady', to be developed
        flag_ignition: TRUE for ignition simulation, stop after ignition
        flag_fast_solver_for_transient: TRUE for fast solver for transient simulation, physical properties would't be updated during the time step
        #TODO: add gas_velocity related simulation for 'Quasi_Steady'
    """
    case_name: str
    fuel_composition: np.ndarray
    droplet_radius: float
    boundary_temperature: float
    initial_pressure: float
    initial_temperature: float
    liquid_cell_count: int
    gas_cell_count: int
    initital_time_step: float
    mechanism_file: str = 'Mech_react.yaml'
    liquid_solver_type: str = 'ITCID'
    gas_solver_type: str = 'react'
    gas_velocity: float = 0.0
    flag_ignition: bool = False
    flag_fast_solver_for_transient: bool = False
    standard_deviation: Optional[float] = None
    I_ini: Optional[float] = None

@dataclass
class TemperatureChangeTracker:
    """combustion reaction temperature change tracker class
    
    Attributes:
        max_temperature_change: first reaction maximum temperature change [K]
        max_temperature_change_location: first reaction maximum temperature change location index
        second_max_temperature_change: second reaction maximum temperature change [K]
        second_max_temperature_change_location: second reaction maximum temperature change location index
        temperature_before_advance: temperature distribution before advance
    """
    max_temperature_change: float = 0.0
    max_temperature_change_location: int = 0
    second_max_temperature_change: float = 0.0
    second_max_temperature_change_location: int = 0
    temperature_before_advance: np.ndarray = None

    def record_before_advance(self, temperatures: np.ndarray) -> None:
        """record temperature distribution before advance"""
        self.temperature_before_advance = temperatures.copy()

    def calculate_after_advance(self, temperatures: np.ndarray, is_second_reaction: bool = False) -> None:
        """calculate temperature change after advance"""
        if self.temperature_before_advance is None:
            self.temperature_before_advance = temperatures.copy()
            return
            
        temperature_changes = np.abs(temperatures - self.temperature_before_advance)
        
        if is_second_reaction:
            self.second_max_temperature_change = np.max(temperature_changes)
            self.second_max_temperature_change_location = np.argmax(temperature_changes)
        else:
            self.max_temperature_change = np.max(temperature_changes)
            self.max_temperature_change_location = np.argmax(temperature_changes)
        

class Simulation:
    """droplet evaporation simulation main class
    
    Features:
    - initialize simulation 
    - execute time advance
    - calculate interface equilibrium
    - update phase state
    """
    def __init__(self, params: SimulationParameters):
        """initialize simulation """
        self.params = params       
        # runtime
        self.runtime: Optional[Runtime] = None  
        self.data_manager: Optional[DataManager] = None
        
        # liquid phase related
        self.liquid_template: Optional[Liquid] = None
        self.liquid_array: Optional[LiquidArray] = None
        self.old_liquid_array: Optional[LiquidArray] = None
        
        # gas phase related
        self.gas_template: Optional[ct.Solution] = None
        self.gas_array: Optional[ct.SolutionArray] = None
        self.old_gas_array: Optional[ct.SolutionArray] = None
        self.gas_reactors: Optional[list[ct.IdealGasConstPressureReactor]] = None
        self.gas_react_nets: Optional[list[ct.ReactorNet]] = None

        
        # grid and interface
        self.grid: Optional[Grid] = None
        self.surface: Optional[Surface] = None
        self.gas_surface: Optional[ct.Solution] = None
        self.liquid_surface: Optional[Liquid] = None
        self.liquid_flux: Optional[Liquid] = None
        self.gas_inf: Optional[ct.Solution] = None
        self.liquid_00: Optional[Liquid] = None

        # temperature tracking
        self.temperature_tracker = TemperatureChangeTracker()
        self.outer_cell_index: Optional[int] = None
        
        # interface convergence tracking
        self.interface_non_convergence_count: int = 0

        # flag
        self.flag_ignition: bool = False
        self.flag_fast_solver_for_transient: bool = False

    def initialize(self):
        """initialize simulation components with the following order:
        1. runtime
        2. data manager
        3. liquid phase state
        4. gas phase state
        5. grid system
        6. interface condition
        7. solver
        """
        self.runtime = Runtime(time_step=self.params.initital_time_step)
        self.flag_ignition = self.params.flag_ignition
        self.flag_fast_solver_for_transient = self.params.flag_fast_solver_for_transient
        # initialize liquid phase template
        self.liquid_template = Liquid(self.params.initial_temperature,self.params.initial_pressure,self.params.fuel_composition)
        
        # initialize liquid_array
        self.liquid_array = LiquidArray(self.liquid_template,self.params.liquid_cell_count)
        self.old_liquid_array = LiquidArray(self.liquid_template,self.params.liquid_cell_count)
        self.liquid_00 = Liquid(self.params.initial_temperature,self.params.initial_pressure,self.params.fuel_composition)
        self.liquid_00.TPX(
            temperature=self.liquid_array.T[0],
            pressure=self.liquid_array.P[0],
            composition=self.liquid_array.X[0]
        )
        
        # initialize droplet parameters
        initial_volume = (1/3) *  self.params.droplet_radius**3
        initial_mass = initial_volume * self.liquid_array.liquids[-1].density_mass
        initial_mass_i = initial_mass * self.liquid_array.liquids[-1].mass_fraction
        self.droplet_params = DropletParameters(
            mass_0=initial_mass,
            volume_0=initial_volume,
            mass_i_0=initial_mass_i.copy(),
            mass=initial_mass,
            volume=initial_volume,
            mass_i=initial_mass_i.copy(),
            mass_new=initial_mass,
            volume_new=initial_volume,
            mass_i_new=initial_mass_i.copy(),
        )
        
        # initialize gas phase template and array
        self.gas_template = ct.Solution(self.params.mechanism_file)
        
        self.gas_array = ct.SolutionArray(self.gas_template, self.params.gas_cell_count)
        self.gas_array.TPX = (
            np.full(self.params.gas_cell_count, self.params.boundary_temperature),
            np.full(self.params.gas_cell_count, self.params.initial_pressure),
            {'N2': 0.79, 'O2': 0.21}
        )
        self.old_gas_array = ct.SolutionArray(self.gas_template, self.params.gas_cell_count)
        self.old_gas_array.TPX = self.gas_array.TPX
        self.gas_inf = ct.Solution(self.params.mechanism_file)
        self.gas_inf.TPX = self.gas_array[-1].TPX

        # initialize surface related objects
        self.gas_surface = ct.Solution(self.params.mechanism_file)
        self.gas_surface.TPX = self.gas_array[0].TPX
        self.liquid_surface = Liquid(
            self.liquid_array.liquids[-1].temperature,
            self.liquid_array.liquids[-1].pressure,
            self.liquid_array.liquids[-1].composition
        )
        self.liquid_flux = Liquid(
            self.liquid_surface.temperature,
            self.liquid_surface.pressure,
            self.liquid_surface.composition
        )
        
        # initialize grid
        self.grid = Grid(
            liquid_array=self.liquid_array,
            old_liquid_array=self.old_liquid_array,
            gas_array=self.gas_array,
            old_gas_array=self.old_gas_array,
            params=GridParameters(
                droplet_radius_init=self.params.droplet_radius,
                droplet_radius=self.params.droplet_radius,
                gas_cell_count=self.params.gas_cell_count,
                liquid_cell_count=self.params.liquid_cell_count
            ),
            runtime=self.runtime,
            liquid_00=self.liquid_00,
            liquid_surface=self.liquid_surface,
            gas_surface=self.gas_surface,
            gas_inf=self.gas_inf
        )
        
        # initialize surface
        self.surface = Surface(
            grid=self.grid,
            gas_surface=self.gas_surface,
            liquid_surface=self.liquid_surface
        )
        self.surface.standard_deviation = self.params.standard_deviation
        self.surface.I_ini = self.params.I_ini
        
        # initialize data_manager
        self.data_manager = DataManager(
            case_name=self.params.case_name,
            grid=self.grid,
            liquid_template=self.liquid_template,
            gas_template=self.gas_template,
            droplet_params=self.droplet_params,
            simulation_params=self.params,
            liquid_gas_save_interval=100,
            appearance_save_interval=10,
            liquid_fuel_indices=self.surface.liquid_fuel_indices
        )

        # steady state initialization
        self.outer_cell_index = self._init_steady_state()
        
        # initialize gas phase reactors and reaction networks (only when gas_solver_type is 'react')
        if self.params.gas_solver_type == 'react':
            self.gas_reactors = []
            self.gas_react_nets = []
            for i in range(self.params.gas_cell_count):
                # create a new Solution object for each grid point
                gas = ct.Solution(self.params.mechanism_file)
                # set the state of the Solution object
                gas.TPX = self.gas_array[i].TPX
                reactor = ct.IdealGasConstPressureReactor(gas)
                self.gas_reactors.append(reactor)
                # create an independent reaction network for each reactor
                react_net = ct.ReactorNet([reactor])
                self.gas_react_nets.append(react_net)
        else:
            self.gas_reactors = None
            self.gas_react_nets = None

        # initialize gas phase solver
        if self.params.gas_solver_type in ['Quasi_Steady']:
            # create gas phase reference state and flux state
            self.gas_ref = ct.Solution(self.params.mechanism_file)
            self.gas_flux = ct.Solution(self.params.mechanism_file)
            # initialize quasi-steady gas phase solver
            self.gas_solver = GasSolverQS(
                grid=self.grid,
                surface=self.surface,
                gas_inf=self.gas_inf,
                gas_ref=self.gas_ref,
                gas_flux=self.gas_flux,
                liquid_flux=self.liquid_flux,
                gas_velocity=self.params.gas_velocity
            )
        else:
            self.gas_solver = GasSolver(
                grid=self.grid,
                surface=self.surface,
                gas_inf=self.gas_inf,
                flag_fast_solver_for_transient=self.flag_fast_solver_for_transient
            )

        # initialize liquid phase solver
        self.liquid_solver = LiquidSolver(
            grid=self.grid,
            liquid_00=self.liquid_00,
            surface=self.surface,
            solver_type=self.params.liquid_solver_type
        )

    def _init_steady_state(self) -> int:
        """steady state initialization
        
        initialize gas phase temperature and composition based on droplet radius and ini_ratio
        return the index of the outermost grid within (ini_ratio*2-1) times the distance
        """
        # calculate the two critical radii inside and outside the boundary layer
        droplet_radius = self.grid.params.droplet_radius
        ini_ratio = 1.05 # ratio of the calculation domain radius to the droplet radius [-]
        r1 = droplet_radius * ini_ratio
        r2 = r1 * 2 - droplet_radius

        # find the grid between r1 and r2
        positions_volume_centers = self.grid.gas_grid.positions_volume_centers
        mask_r1 = positions_volume_centers <= r1
        mask_r2 = (positions_volume_centers > r1) & (positions_volume_centers <= r2)

        # get the index of the outermost grid
        outer_cell_index = np.where(mask_r2)[0][-1] if np.any(mask_r2) else -1

        # initialize temperature and composition
        if np.any(mask_r1):
            # use the surface value for the grid in the range of r1
            self.gas_array[mask_r1].TPY = (
                self.surface.gas_surface.T,
                self.surface.gas_surface.P,
                self.surface.gas_surface.Y
            )

        if np.any(mask_r2):
            # use the inverse proportional function interpolation for the grid in the range of r1 to r2
            x = positions_volume_centers[mask_r2]
            x0, T0 = r1, self.surface.gas_surface.T
            x1, T1 = r2, self.gas_inf.T

            # construct the inverse proportional function T = a/x + b
            A = np.array([[1/x0, 1], [1/x1, 1]])
            b_vec = np.array([T0, T1])
            a, b = np.linalg.solve(A, b_vec)
            T_interp = a/x + b

            # use the inverse proportional function interpolation for each component
            n_species = len(self.surface.gas_surface.Y)
            Y_interp = np.zeros((len(x), n_species))
            for i in range(n_species):
                Y0 = self.surface.gas_surface.Y[i]
                Y1 = self.gas_inf.Y[i]
                b_vec = np.array([Y0, Y1])
                a, b = np.linalg.solve(A, b_vec)
                Y_interp[:, i] = a/x + b

            # update the state of the gas phase array
            self.gas_array[mask_r2].TPY = (
                T_interp,
                self.surface.gas_surface.P,
                Y_interp
            )
        # synchronize the old_gas_array
        self.grid.old_gas_array.TPY = self.gas_array.TPY
        return outer_cell_index
        
    def run(self):
        """execute the main loop of the simulation"""
        if self.params.liquid_solver_type in ['ITCID', 'FTCID', 'ITCFD', 'FTCFD']:
            self._run_unified_loop()
        else:
            raise ValueError(f"unsupported solver type: {self.params.liquid_solver_type}")

    def _run_unified_loop(self):
        """unified solver loop"""
        # initialize the temperature change tracker (if needed and gas_solver_type == 'react')
        if self.params.gas_solver_type == 'react':
            self.temperature_tracker.record_before_advance(self.gas_array.T)
            print("=== the temperature change tracker is initialized ===")
            print("initialize the temperature change tracker (only when gas_solver_type == 'react')")
            print("="*50)
        time_step_initial = self.runtime.time_step
        while self.runtime.is_running():
            # update the time step
            self.runtime.advance()
            
            # update the gas phase surface state
            self.gas_surface.TPY = self.gas_surface.T, self.gas_surface.P, self.gas_array[0].Y
            self.gas_surface.TPX = self.gas_surface.T, self.gas_surface.P, self.surface.calculate_gas_surface_composition()
            
            # update the grid
            self.grid.reset_grid()

            # perform the first chemical reaction (if gas_solver_type == 'react')
            if self.params.gas_solver_type == 'react':
                self._perform_chemical_reaction()
                if self.temperature_tracker.max_temperature_change > 1 or self.temperature_tracker.second_max_temperature_change > 1:
                    self.runtime.time_step = time_step_initial/10
            # calculate the interface equilibrium and temperature, and advance the gas phase time step
            # the gas_array_iter of the gas phase and the liquid_array_iter.T of the liquid phase are the latest
            self._compute_interface_equilibrium()
            
            # update the gas phase state 
            self.gas_array.TPX = self.gas_solver.gas_array_iter.T, self.gas_solver.gas_array_iter.P, self.gas_solver.gas_array_iter.X
            
            # update the liquid phase state and droplet parameters (through the auxiliary method to handle the difference)
            self._update_liquid_phase_and_volume()
            
            # save all data 
            self.data_manager.save_all(self)
            
            # perform the second chemical reaction (if gas_solver_type == 'react')
            if self.params.gas_solver_type == 'react':
                self._perform_chemical_reaction(is_second_reaction=True)
            
            # add iteration count
            self.data_manager.increment_iteration()
            if self.flag_ignition:
                if self.gas_array.T.max() > 2000:
                    print("=== droplet ignition performance ===")
                    print(f"* max temperature ({self.gas_array.T.max():.10f}) is greater than 2000K, simulation ends.")
                    print(f"* ignition time is {self.runtime.current_time:.10f}s")
                    print(f"* normalized ignition time is {self.runtime.current_time/self.params.droplet_radius**2:.10f}s/m^2")
                    print(f"* ignition location is {self.grid.gas_grid.positions_volume_centers[self.gas_array.T.argmax()]:.10f}m")
                    print("="*50)
                    self.runtime.stop()
                    break
            # check if the droplet radius is less than 1/10 of the initial radius, if so, end the simulation
            if self.grid.params.droplet_radius < 0.1 * self.params.droplet_radius:
                print("=== droplet evaporation performance ===")
                print(f"* droplet radius ({self.grid.params.droplet_radius:.10f}) is less than 1/10 of the initial radius ({self.params.droplet_radius:.10f}), simulation ends.")
                print("="*50)
                self.runtime.stop()
                break
        
        print(f"\nsimulation ends, interface temperature non-convergence count: {self.interface_non_convergence_count}")
        print("="*50)

    def _perform_chemical_reaction(self, is_second_reaction: bool = False) -> None:
        """perform chemical reaction
        
        steps:
        1. sync state to reactor
        2. record temperature
        3. perform reaction
        4. sync result
        5. calculate temperature change
        """
        # 1. sync gas_array to reactor state and re-initialize the reaction network
        for i, (reactor, react_net) in enumerate(zip(self.gas_reactors, self.gas_react_nets)):
            # update the reactor state from gas_array
            reactor.thermo.TPX = self.gas_array[i].TPX
            reactor.syncState() # Ensure reactor internal state is consistent after TPX update
        # 2. record the temperature before advance
        self.temperature_tracker.record_before_advance(self.gas_array.T)
        
        # 3. perform reaction for all reactors
        target_time = self.runtime.current_time if is_second_reaction else self.runtime.current_time - self.runtime.time_step / 2
        for react_net in self.gas_react_nets:
            react_net.advance(target_time)
        
        # 4. sync reactor state to gas_array
        for i, reactor in enumerate(self.gas_reactors):
            self.gas_array[i].TPX = reactor.thermo.TPX
        
        # 5. calculate the temperature change after advance
        self.temperature_tracker.calculate_after_advance(self.gas_array.T, is_second_reaction)
        if is_second_reaction:
            print("=== Combustion reaction performance ===")
            print(f"* GAS React: max temperature change in second dt/2: {self.temperature_tracker.second_max_temperature_change:.5f}K, location: {self.temperature_tracker.second_max_temperature_change_location},ratio:{self.grid.gas_grid.positions_volume_centers[self.temperature_tracker.second_max_temperature_change_location]/self.grid.params.droplet_radius:.2f}")
        else:
            print(f"* GAS React: max temperature change in first dt/2: {self.temperature_tracker.max_temperature_change:.5f}K, location: {self.temperature_tracker.max_temperature_change_location},ratio:{self.grid.gas_grid.positions_volume_centers[self.temperature_tracker.max_temperature_change_location]/self.grid.params.droplet_radius:.2f}")
            print("="*50+"\n")

    def _compute_interface_equilibrium_transient(self):
        """
        calculate the gas-liquid interface equilibrium (for evap and react solver)
        
        this method uses iterative method to solve the interface temperature, main steps:
        1. update the liquid phase and gas surface state
        2. solve the gas phase temperature and composition
        3. calculate the new interface temperature
        4. check the convergence
        """
        max_iter = 5
        tolerance = 1e-4
        T_interface = self.surface.state.temperature
        gas_stefan_velocity = 0.5 * (self.surface.state.stefan_velocity_gas + self.surface.state.old_stefan_velocity_gas)
        interface_params = {} 
        self.gas_solver.gas_array_iter.TPX = self.gas_array.TPX
        print("=== Interface equilibrium iteration ===")
        for iter_idx in range(max_iter):
            print(f"\n* begin  {iter_idx+1} temperature teration({self.params.liquid_solver_type}):")
            # update the liquid phase surface state
            self.liquid_surface.TPY(T_interface, self.liquid_array.P[-1], self.liquid_array.Y[-1])
            # update the gas phase surface state
            self.gas_surface.TPX = T_interface, self.liquid_surface.pressure, self.surface.calculate_gas_surface_composition()
            # calculate the gas phase relative velocity
            self.calculate_gas_relative_velocity(gas_stefan_velocity)
            # solve the gas phase temperature
            self.gas_solver.gas_array_iter.TPX = self.gas_array.TPX
            result_T_gas = self.gas_solver.T_solver.solve_bdf(Dt=self.runtime.time_step, T0=self.gas_array.T)
            T_new_gas_field = result_T_gas.y.T[-1]
            self.gas_solver.gas_array_iter.TPX = T_new_gas_field, self.gas_array.P, self.gas_array.X
            # calculate the gas phase relative velocity after T
            self.calculate_gas_relative_velocity_afterT(gas_stefan_velocity)
            # solve the gas phase composition
            result_y_gas = self.gas_solver.y_solver.solve_bdf(Dt=self.runtime.time_step, y0=self.gas_array.Y[:, 1:].reshape(-1, order='F'))
            y_new_gas_flat = result_y_gas.y.T[-1, :]
            y_new_gas_2d = y_new_gas_flat.reshape(self.gas_array.Y.shape[0], self.gas_array.Y.shape[1]-1, order='F')
            # add N2 concentration at the front (1 - sum of other species)
            n2_concentration = 1.0 - np.sum(y_new_gas_2d, axis=1, keepdims=True)
            y_new_gas_2d = np.concatenate([n2_concentration, y_new_gas_2d], axis=1)
            y_new_gas_2d = y_new_gas_2d / y_new_gas_2d.sum(axis=1, keepdims=True)
            self.gas_solver.gas_array_iter.TPY = self.gas_solver.gas_array_iter.T, self.gas_solver.gas_array_iter.P, y_new_gas_2d

            # calculate the new interface temperature and interface parameters
            T_interface_new, interface_params = self._calculate_interface()

            temperature_error = (T_interface_new - T_interface) / T_interface
            print(f"* interface temperature after iteration: {T_interface_new:.4f}, relative error: {temperature_error:.4%}")
            T_interface = 0.3 * T_interface + 0.7 * T_interface_new
            if T_interface >= self.gas_solver.gas_array_iter.T[0]:
                print(f"* interface temperature ({T_interface:.4f}) is greater than gas phase temperature ({self.gas_solver.gas_array_iter.T[0]:.4f}), iteration interface temperature to gas phase temperature")
                T_interface = self.gas_solver.gas_array_iter.T[0]
            if -1*tolerance < temperature_error < tolerance:
                print(f"\n* interface temperature converges after {iter_idx+1} iterations, relative error: {temperature_error:.4%}")
                break
        else:
            print(f"\n* interface temperature does not converge after {max_iter} iterations, relative error: {temperature_error:.4%}")
            self.interface_non_convergence_count += 1
        print("="*50+"\n")
        # update the droplet parameters and gas-liquid interface state
        self._update_interface_state(T_interface, interface_params)

    def _compute_interface_equilibrium_QS(self):
        """calculate the gas-liquid interface equilibrium under quasi-steady state
        """
        QS_calc_params = self.gas_solver.update_calc_params()
        evaporation_rate_liquid = np.zeros_like(self.liquid_surface.mass_fraction)
        evaporation_rate_liquid[self.surface.liquid_fuel_indices] = QS_calc_params.E_i * QS_calc_params.mdot
        new_mass_liquid = self.droplet_params.mass - QS_calc_params.mdot * self.runtime.time_step
        new_mass_i_liquid = self.droplet_params.mass_i - evaporation_rate_liquid * self.runtime.time_step
        interface_diffusion_flux_gas = - QS_calc_params.rho_ref * QS_calc_params.D_ref * QS_calc_params.dys_dr
        stefan_velocity_gas = interface_diffusion_flux_gas/ (QS_calc_params.rho_ref * (1 - QS_calc_params.y_fs))
        stefan_velocity_liquid = np.sum(evaporation_rate_liquid / self.liquid_flux.density_mass_ij())/self.grid.params.droplet_radius**2
        heat_rate_liquid = -QS_calc_params.Q_dot
        heat_flux_liquid = heat_rate_liquid / (self.grid.params.droplet_radius ** 2)
        heat_flux_gas = -QS_calc_params.Q_gas / (self.grid.params.droplet_radius ** 2)
        evaporation_heat_flux = -QS_calc_params.Q_hv / (self.grid.params.droplet_radius ** 2)
        last_temperature_original = self.liquid_array[-1].temperature
        if self.params.liquid_solver_type == 'ITCID' or self.params.liquid_solver_type == 'ITCFD':
            heat_capacity = self.liquid_array[-1].cp_mass * self.droplet_params.mass
            delta_T_interface_rate = -heat_rate_liquid / heat_capacity 
            T_interface_new = last_temperature_original + delta_T_interface_rate * self.runtime.time_step
        elif self.params.liquid_solver_type == 'FTCID' or self.params.liquid_solver_type == 'FTCFD':
            self.liquid_solver.liquid_array_iter.TPY(self.liquid_array.T, self.liquid_array.P, self.liquid_array.Y)
            result_T_liquid = self.liquid_solver.T_solver.solve_bdf(Dt=self.runtime.time_step, T0=self.liquid_array.T,heat_rate_liquid=heat_rate_liquid)
            T_new_liquid_field = result_T_liquid.y.T[-1]
            self.liquid_solver.liquid_array_iter.TPY(T_new_liquid_field, self.liquid_solver.liquid_array_iter.P, self.liquid_solver.liquid_array_iter.Y)
            T_interface_new = self.liquid_solver.liquid_array_iter[-1].temperature
        interface_params = {
            "new_mass_liquid": new_mass_liquid,
            "new_mass_i_liquid": new_mass_i_liquid,
            "stefan_velocity_gas": stefan_velocity_gas,
            "stefan_velocity_liquid": stefan_velocity_liquid,
            "interface_diffusion_flux_gas": interface_diffusion_flux_gas,
            "evaporation_rate_liquid": evaporation_rate_liquid,
            "heat_flux_liquid": heat_flux_liquid,
            "heat_flux_gas": heat_flux_gas,
            "evaporation_heat_flux": evaporation_heat_flux
        }
        self._update_interface_state(T_interface_new, interface_params)
    def _update_interface_state(self, T_interface, interface_params):
        """
        更新界面状态和液滴参数
        
        参数:
            T_interface (float): 界面温度
            interface_params (dict): 界面参数
        """
        self.liquid_surface.TPY(T_interface, self.liquid_array.P[-1], self.liquid_array.Y[-1])
        self.gas_surface.TPX = T_interface, self.liquid_surface.pressure, self.surface.calculate_gas_surface_composition()

        self.droplet_params.mass_new = interface_params["new_mass_liquid"]
        self.droplet_params.mass_i_new = interface_params["new_mass_i_liquid"]

        old_stefan_velocity_gas = self.surface.state.stefan_velocity_gas
        old_stefan_velocity_liquid = self.surface.state.stefan_velocity_liquid
        self.surface.state = type(self.surface.state)(
            temperature=T_interface,
            stefan_velocity_liquid=interface_params["stefan_velocity_liquid"],
            stefan_velocity_gas=interface_params["stefan_velocity_gas"],
            old_stefan_velocity_liquid=old_stefan_velocity_liquid,
            old_stefan_velocity_gas=old_stefan_velocity_gas,
            interface_diffusion_flux_liquid=0,
            interface_diffusion_flux_gas=interface_params["interface_diffusion_flux_gas"].copy(),
            evaporation_rate=interface_params["evaporation_rate_liquid"],
            heat_flux_liquid=interface_params["heat_flux_liquid"],
            heat_flux_gas=interface_params["heat_flux_gas"],
            evaporation_heat_flux=interface_params["evaporation_heat_flux"]
        )

    def _compute_interface_equilibrium(self):
        """
        select the appropriate interface equilibrium calculation method according to the gas phase solver type
        """
        if self.params.gas_solver_type in ['evap', 'react']:
            self._compute_interface_equilibrium_transient()
        elif self.params.gas_solver_type in ['Quasi_Steady']:
            self._compute_interface_equilibrium_QS()
        else:
            raise ValueError(f"Unsupported gas phase solver type: {self.params.gas_solver_type}")

    def calculate_gas_relative_velocity(self, gas_stefan_velocity):
        """calculate the gas phase relative velocity"""
        left_areas = self.grid.gas_grid.areas_left_boundary
        left_density = np.concatenate([[self.gas_surface.density_mass], self.gas_array.density_mass[:-1]])
        right_areas = self.grid.gas_grid.areas_right_boundary
        right_density = self.gas_array.density_mass
        rel_vel = np.zeros(self.grid.gas_grid.cell_count+1)
        # 先设置第一个网格
        rel_vel[0] = gas_stefan_velocity
        self.gas_solver._relative_velocity[0] = gas_stefan_velocity
        # 计算递推因子
        factors = left_areas * left_density / (right_areas * right_density)
        # 累乘得到所有网格的相对速度
        rel_vel[1:] = rel_vel[0] * np.cumprod(factors)
        self.gas_solver.set_velocity(rel_vel[1:])

    def calculate_gas_relative_velocity_afterT(self, gas_stefan_velocity):
        """calculate the gas phase relative velocity"""
        left_areas = self.grid.gas_grid.areas_left_boundary
        left_density = np.concatenate([[self.gas_surface.density_mass], self.gas_solver.gas_array_iter.density_mass[:-1]])
        right_areas = self.grid.gas_grid.areas_right_boundary
        right_density = self.gas_solver.gas_array_iter.density_mass
        rel_vel = np.zeros(self.grid.gas_grid.cell_count+1)
        # set the first grid
        rel_vel[0] = gas_stefan_velocity
        self.gas_solver._relative_velocity[0] = gas_stefan_velocity
        # calculate the recursive factor
        factors = left_areas * left_density / (right_areas * right_density)
        # multiply to get the relative velocity of all grids
        rel_vel[1:] = rel_vel[0] * np.cumprod(factors)
        self.gas_solver.set_velocity(rel_vel[1:])

    def _calculate_interface(self) -> tuple[float, dict]:
        """calculate the gas-liquid interface equilibrium
        
        Returns:
            - new interface temperature
            - interface parameter update dictionary
        """
        # create the fuel and non-fuel mask
        gas_non_fuel_mask = np.zeros_like(self.gas_surface.Y, dtype=bool)
        gas_non_fuel_mask[self.surface.gas_non_fuel_indices] = True
        gas_fuel_mask = ~gas_non_fuel_mask
        liquid_fuel_indices = self.surface.liquid_fuel_indices
        
        # 1. calculate the physical parameters
        gas_dr = self.grid.gas_grid.positions_volume_centers[0] - self.grid.gas_grid.positions_left_boundary[0]
        D_gas = 0.5 * (self.gas_surface.mix_diff_coeffs_mass + self.gas_solver.gas_array_iter[0].mix_diff_coeffs_mass)
        rho_gas = 0.5 * (self.gas_surface.density_mass + self.gas_solver.gas_array_iter[0].density_mass)
        thermal_conductivity_gas_interface = 0.5 * (self.gas_surface.thermal_conductivity + self.gas_solver.gas_array_iter[0].thermal_conductivity)
        self.liquid_flux.temperature = self.gas_surface.T

        # 2. solve the species conservation condition and the change of droplet mass according to the interface temperature condition
        # 2.1 calculate the gas phase diffusion and stefan velocity
        interface_diffusion_flux_gas = -rho_gas * D_gas * (self.gas_solver.gas_array_iter[0].Y - self.gas_surface.Y) / gas_dr
        interface_diffusion_flux_gas_fuel = interface_diffusion_flux_gas[gas_fuel_mask]
        interface_diffusion_flux_liquid_fuel = np.zeros(self.liquid_template.mass_fraction.shape[0])
        interface_diffusion_flux_liquid_fuel[liquid_fuel_indices] = interface_diffusion_flux_gas_fuel.copy()
        stefan_velocity_gas = -np.sum(interface_diffusion_flux_gas[gas_non_fuel_mask]) / (self.gas_surface.density_mass * np.sum(self.gas_surface.Y[gas_non_fuel_mask])) 
        evaporation_flux_gas = stefan_velocity_gas * self.gas_surface.density_mass * self.gas_surface.Y[gas_fuel_mask] + interface_diffusion_flux_gas_fuel
        evaporation_flux_liquid = np.zeros(self.liquid_template.mass_fraction.shape[0])
        evaporation_flux_liquid[liquid_fuel_indices] = evaporation_flux_gas.copy()
        stefan_velocity_liquid = np.sum(evaporation_flux_liquid/ self.liquid_flux.density_mass_ij())
        evaporation_rate_liquid = evaporation_flux_liquid * self.grid.params.droplet_radius ** 2
        # 2.2 calculate the change of droplet mass
        delta_mass_rate_liquid_scalar = np.sum(evaporation_rate_liquid) 
        new_mass_liquid = self.droplet_params.mass - delta_mass_rate_liquid_scalar * self.runtime.time_step
        delta_mass_i_liquid_step = evaporation_rate_liquid * self.runtime.time_step
        new_mass_i_liquid = self.droplet_params.mass_i - delta_mass_i_liquid_step

        # 3. calculate the heat transfer according to the mass transfer calculation situation
        heat_flux_gas = -thermal_conductivity_gas_interface * (self.gas_solver.gas_array_iter[0].T - self.gas_surface.T) / gas_dr
        evaporation_heat_flux = evaporation_flux_liquid @ self.liquid_flux.heat_vaporization_mass_ij()
        heat_flux_liquid = heat_flux_gas + evaporation_heat_flux    
        heat_rate_liquid = heat_flux_liquid * self.grid.params.droplet_radius ** 2
        
        # 4. calculate the new interface temperature according to the heat transfer situation
        last_temperature_original = self.liquid_array[-1].temperature
        if self.params.liquid_solver_type == 'ITCID' or self.params.liquid_solver_type == 'ITCFD':
            heat_capacity = self.liquid_array[-1].cp_mass * self.droplet_params.mass
            delta_T_interface_rate = -heat_rate_liquid / heat_capacity 
            T_interface_new = last_temperature_original + delta_T_interface_rate * self.runtime.time_step
        elif self.params.liquid_solver_type == 'FTCID' or self.params.liquid_solver_type == 'FTCFD':
            self.liquid_solver.liquid_array_iter.TPY(self.liquid_array.T, self.liquid_array.P, self.liquid_array.Y)
            result_T_liquid = self.liquid_solver.T_solver.solve_bdf(Dt=self.runtime.time_step, T0=self.liquid_array.T,heat_rate_liquid=heat_rate_liquid)
            T_new_liquid_field = result_T_liquid.y.T[-1]
            self.liquid_solver.liquid_array_iter.TPY(T_new_liquid_field, self.liquid_solver.liquid_array_iter.P, self.liquid_solver.liquid_array_iter.Y)
            T_interface_new = self.liquid_solver.liquid_array_iter[-1].temperature
        
        # 5. integrate all parameters and return
        required_params = {
            "new_mass_liquid": new_mass_liquid,
            "new_mass_i_liquid": new_mass_i_liquid,
            "stefan_velocity_gas": stefan_velocity_gas,
            "stefan_velocity_liquid": stefan_velocity_liquid,
            "interface_diffusion_flux_gas": interface_diffusion_flux_gas,
            "evaporation_rate_liquid": evaporation_rate_liquid,
            "heat_flux_liquid": heat_flux_liquid,
            "heat_flux_gas": heat_flux_gas,
            "evaporation_heat_flux": evaporation_heat_flux
        }
        
        return T_interface_new, required_params

    def _update_liquid_phase_and_volume(self):
        """update the liquid phase state and droplet volume"""
        liquid_mass_fraction = self.droplet_params.mass_i_new / self.droplet_params.mass_i_new.sum()
        liquid_mass_fraction_array = np.tile(liquid_mass_fraction, (self.params.liquid_cell_count, 1))
        temperature_array = np.full(self.liquid_array.n_grid, self.surface.state.temperature)
        liquid_mass_cell = self.liquid_array.density_mass * self.grid.liquid_grid.volumes
        # update the liquid phase mass transfer
        if self.params.liquid_solver_type == 'ITCID': 
            self.liquid_array.TPY(temperature_array, self.liquid_array.P, liquid_mass_fraction_array)
        elif self.params.liquid_solver_type == 'FTCID':
            self.liquid_array.TPY(self.liquid_solver.liquid_array_iter.T, self.liquid_array.P, liquid_mass_fraction_array)
        elif self.params.liquid_solver_type == 'ITCFD':
            self.liquid_solver.liquid_array_iter.TPY(temperature_array, self.liquid_array.P, self.liquid_array.Y)
            result_y_liquid = self.liquid_solver.y_solver.solve_bdf(
                Dt=self.runtime.time_step,
                y0=self.liquid_array.Y.reshape(-1, order='F'),
                mass_rate_liquid=self.surface.state.evaporation_rate
            )
            y_new_liquid_flat = result_y_liquid.y.T[-1, :]
            y_new_liquid_2d = y_new_liquid_flat.reshape(self.liquid_array.Y.shape, order='F')
            y_new_liquid_2d = y_new_liquid_2d / y_new_liquid_2d.sum(axis=1, keepdims=True)
            self.liquid_array.TPY(temperature_array,self.liquid_array.P,y_new_liquid_2d)
        elif self.params.liquid_solver_type == 'FTCFD' :
            self.liquid_solver.liquid_array_iter.TPY(self.liquid_solver.liquid_array_iter.T,self.liquid_array.P,self.liquid_array.Y)
            result_y_liquid = self.liquid_solver.y_solver.solve_bdf(
                Dt=self.runtime.time_step,
                y0=self.liquid_array.Y.reshape(-1, order='F'),
                mass_rate_liquid=self.surface.state.evaporation_rate
            )
            y_new_liquid_flat = result_y_liquid.y.T[-1, :]
            y_new_liquid_2d = y_new_liquid_flat.reshape(self.liquid_array.Y.shape, order='F')
            y_new_liquid_2d = y_new_liquid_2d / y_new_liquid_2d.sum(axis=1, keepdims=True)
            self.liquid_array.TPY(self.liquid_solver.liquid_array_iter.T,self.liquid_array.P,y_new_liquid_2d)
        liquid_cell_volumes = liquid_mass_cell / self.liquid_array.density_mass
        liquid_cell_volumes[-1] = liquid_cell_volumes[-1] -np.sum(self.surface.state.evaporation_rate)*self.runtime.time_step/self.liquid_array.density_mass[-1]

        liquid_right_boundaries = np.cbrt(3 * np.cumsum(liquid_cell_volumes))
        liquid_left_boundaries = np.concatenate([[0.0], liquid_right_boundaries[:-1]])
        liquid_volume_centers = 3 * (liquid_right_boundaries**4 - liquid_left_boundaries**4) / \
                               (4 * (liquid_right_boundaries**3 - liquid_left_boundaries**3))
        
        # update the liquid phase grid geometry in the grid
        self.grid.liquid_grid.volumes = liquid_cell_volumes.copy()
        self.grid.liquid_grid.positions_right_boundary = liquid_right_boundaries.copy()
        self.grid.liquid_grid.positions_left_boundary = liquid_left_boundaries.copy()
        self.grid.liquid_grid.positions_volume_centers = liquid_volume_centers.copy()
        new_radius = liquid_right_boundaries[-1].copy()
        self.droplet_params.volume_new = np.sum(liquid_cell_volumes)
        # calculate the various units and normalized values
        time_ms = self.runtime.current_time * 1000  # convert to milliseconds
        radius_mm = new_radius * 1000  # convert to millimeters
        radius_change_mm = (new_radius - self.grid.params.droplet_radius) * 1000  # radius change, millimeters
        initial_diameter_mm = self.params.droplet_radius * 2 * 1000  # initial diameter, millimeters
        normalized_time = self.runtime.current_time / (initial_diameter_mm ** 2)  # normalized time: seconds/millimeters²
        current_diameter_mm = new_radius * 2 * 1000  # current diameter, millimeters
        normalized_diameter = (current_diameter_mm ** 2)  / (initial_diameter_mm ** 2)  # 归一化直径：D²/D₀²

        
        print("=== droplet volume and radius ===")
        print(f"time: {time_ms:.3f}ms, radius: {radius_mm:.6f}mm, radius change: {radius_change_mm:.3g}mm, "
              f"\nnormalized time: {normalized_time:.6e}, normalized diameter: {normalized_diameter:.6f}")
        
        # update the droplet parameters
        self.droplet_params.mass = self.droplet_params.mass_new
        self.droplet_params.volume = self.droplet_params.volume_new
        self.droplet_params.mass_i = self.droplet_params.mass_i_new.copy()
        
        # update the grid parameters
        self.grid.params.interface_motion_velocity = (new_radius - self.grid.params.droplet_radius) / self.runtime.time_step
        self.grid.params.droplet_radius = new_radius

        # update the other boundary conditions
        self.liquid_00.TPY(self.liquid_array.T[0],self.liquid_array.P[0],self.liquid_array.Y[0])
        self.liquid_surface.TPY(self.liquid_array.T[-1],self.liquid_array.P[-1],self.liquid_array.Y[-1])
