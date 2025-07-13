"""
data manager module

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.
"""

import csv
import os
import numpy as np

class DataManager:
    def __init__(self, case_name: str, grid, liquid_template, gas_template, droplet_params, simulation_params, liquid_gas_save_interval=100, appearance_save_interval=10, liquid_fuel_indices=None):
        """
        initialize the data manager
        
        Args:
            case_name: case name
            grid: grid object
            liquid_template: liquid phase template object
            gas_template: gas phase template object
            droplet_params: droplet parameters object
            simulation_params: simulation parameters object
            liquid_gas_save_interval: liquid phase and gas phase data save interval, default is 100
            appearance_save_interval: droplet appearance data save interval, default is 10
        """
        self.case_name = case_name
        self.grid = grid
        self.liquid_template = liquid_template
        self.gas_template = gas_template
        self.droplet_params = droplet_params
        self.simulation_params = simulation_params
        self.iteration_count = 0
        self.liquid_gas_save_interval = liquid_gas_save_interval
        self.appearance_save_interval = appearance_save_interval
        self.liquid_fuel_indices = liquid_fuel_indices
        self.result_dir = os.path.join("result", case_name)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 1. liquid phase state
        self.liquid_species_names = liquid_template.species_names
        self.liquid_header = ['Time', 'Cell_Index', 'Position', 'Temperature'] + self.liquid_species_names
        self.liquid_file = open(os.path.join(self.result_dir, f"liquid-{case_name}.csv"), "w", newline='')
        csv.writer(self.liquid_file).writerow(self.liquid_header)
        self.liquid_file.flush()

        # 2. gas phase state
        self.gas_species_names = gas_template.species_names
        self.gas_header = ['Time', 'Cell_Index', 'Position', 'Temperature'] + self.gas_species_names
        self.gas_file = open(os.path.join(self.result_dir, f"gas-{case_name}.csv"), "w", newline='')
        csv.writer(self.gas_file).writerow(self.gas_header)
        self.gas_file.flush()

        # 3. droplet appearance
        self.appearance_header = [
            'Time', 'DropletRadius', 'D2_time', 'D2_d', 'K_diff',
            'K_cal',  # calculated K value
            'InterfaceVelocity', 'DropletMass', 'SurfaceTemp', 
            'StefanVelocityLiquid', 'StefanVelocityGas',
            'HeatFluxLiquid', 'HeatFluxGas', 'EvaporationHeatFlux',
            'FirstReactionMaxTempChange', 'FirstReactionMaxTempChangeLocation',  # first reaction temperature change
            'SecondReactionMaxTempChange', 'SecondReactionMaxTempChangeLocation',  # second reaction temperature change
            'MaxGasTemperature'  # maximum gas temperature
        ] + [f"EvaporationRate_{name}" for name in self.liquid_species_names] # 40-dimensional liquid phase evaporation rate
        self.appearance_file = open(os.path.join(self.result_dir, f"appearance-{case_name}.csv"), "w", newline='')
        csv.writer(self.appearance_file).writerow(self.appearance_header)
        self.appearance_file.flush()

        # 4. droplet component mass
        self.mass_header = ['Time'] + [f"Mass_{name}" for name in self.liquid_species_names]
        self.mass_file = open(os.path.join(self.result_dir, f"droplet_mass-{case_name}.csv"), "w", newline='')
        csv.writer(self.mass_file).writerow(self.mass_header)
        self.mass_file.flush()

        # 5. quasi-steady gas phase parameters (only when the gas phase solver type is 'Quasi_Steady')
        if simulation_params.gas_solver_type == 'Quasi_Steady':
            self.quasi_steady_header = [
                'Time', 'y_fs', 'B_M', 'B_T', 'Le', 'Q_dot', 'Q_gas', 'Q_hv', 
                'mdot', 'D_ref', 'dys_dr', 'rho_ref'
            ] + [f"E_i_{name}" for name in np.array(self.liquid_species_names)[liquid_fuel_indices]]  # 40-dimensional liquid phase evaporation rate
            self.quasi_steady_file = open(os.path.join(self.result_dir, f"quasi_steady-{case_name}.csv"), "w", newline='')
            csv.writer(self.quasi_steady_file).writerow(self.quasi_steady_header)
            self.quasi_steady_file.flush()
        else:
            self.quasi_steady_file = None

        # print the initialization parameters
        self._print_initialization_parameters()

    def _print_initialization_parameters(self):
        """print the initialization parameters of the data manager"""
        print("\n=== the initialization parameters of the data manager ===")
        print(f"    case name: {self.case_name}")
        print(f"    liquid and gas save interval: {self.liquid_gas_save_interval}")
        print(f"    appearance save interval: {self.appearance_save_interval}")
        print(f"    result directory: {self.result_dir}")
        print(f"    initial pressure: {self.simulation_params.initial_pressure/1e5:.2f} bar")
        print(f"    initial temperature: {self.simulation_params.initial_temperature:.2f} K")
        print(f"    boundary temperature: {self.simulation_params.boundary_temperature:.2f} K")
        print(f"    droplet initial radius: {self.simulation_params.droplet_radius*1000:.2f} mm")
        print(f"    liquid phase solver type: {self.simulation_params.liquid_solver_type}")
        print(f"    gas phase solver type: {self.simulation_params.gas_solver_type}")
        print(f"    flag for ignition: {self.simulation_params.flag_ignition}")
        print(f"    flag for fast solver for transient: {self.simulation_params.flag_fast_solver_for_transient}")
        print("="*50+"\n")

    def save_liquid(self, time, liquid_array, grid):
        if self.iteration_count % self.liquid_gas_save_interval == 0:  
            writer = csv.writer(self.liquid_file)
            for i in range(grid.liquid_grid.cell_count):
                row = [time, i, grid.liquid_grid.positions_volume_centers[i], liquid_array.T[i]]
                row += list(liquid_array.X[i])
                writer.writerow(row)

    def save_gas(self, time, gas_array, grid):
        if self.iteration_count % self.liquid_gas_save_interval == 0:  
            writer = csv.writer(self.gas_file)
            for i in range(grid.gas_grid.cell_count):
                row = [time, i, grid.gas_grid.positions_volume_centers[i], gas_array.T[i]]
                row += list(gas_array.X[i])
                writer.writerow(row)

    def save_appearance(self, time, grid, simulation, surface):
        # add save interval check
        if self.iteration_count % self.appearance_save_interval != 0:
            return
        state = surface.state
        d2_time = time / (simulation.params.droplet_radius*1000 * 2) ** 2
        d2_d = (grid.params.droplet_radius / simulation.params.droplet_radius) ** 2
        
        # calculate K (slope)
        if not hasattr(self, '_last_d2_time'):
            self._last_d2_time = d2_time
            self._last_d2_d = d2_d
            k_diff = 0.0
        else:
            if d2_time != self._last_d2_time:
                k_diff = (d2_d - self._last_d2_d) / (d2_time - self._last_d2_time)
            else:
                k_diff = 0.0
            self._last_d2_time = d2_time
            self._last_d2_d = d2_d

        max_gas_temp = max(simulation.gas_array.T)
        k_cal = 8 * grid.params.droplet_radius * grid.params.interface_motion_velocity * 1E6
        row = [time,grid.params.droplet_radius,d2_time,d2_d,k_diff,k_cal,
               grid.params.interface_motion_velocity,simulation.droplet_params.mass,
               state.temperature,state.stefan_velocity_liquid,state.stefan_velocity_gas,
               state.heat_flux_liquid,state.heat_flux_gas,state.evaporation_heat_flux,
               simulation.temperature_tracker.max_temperature_change,simulation.temperature_tracker.max_temperature_change_location,simulation.temperature_tracker.second_max_temperature_change,simulation.temperature_tracker.second_max_temperature_change_location,max_gas_temp] + list(state.evaporation_rate)  # 40-dimensional liquid phase evaporation rate
        writer = csv.writer(self.appearance_file)
        writer.writerow(row)
        print("=== appearance data ===")
        print(f"K_diff: {k_diff:.3g} mm2/s, K_cal: {k_cal:.3g} mm2/s, interface velocity: {grid.params.interface_motion_velocity:.3g} m/s, droplet mass: {simulation.droplet_params.mass:.3g} kg, interface temperature: {state.temperature:.2f} K, \nliquid heat flux: {state.heat_flux_liquid:.2f} W/m2, gas heat flux: {state.heat_flux_gas:.2f} W/m2, evaporation heat flux: {state.evaporation_heat_flux:.2f} W/m2, maximum gas temperature: {max_gas_temp:.2f} K")
        
        # print the reaction related information when the gas phase solver type is 'react'
        if simulation.params.gas_solver_type == 'react':
            print(f"first half step: {simulation.temperature_tracker.max_temperature_change:.1f}K@{simulation.temperature_tracker.max_temperature_change_location}, second half step: {simulation.temperature_tracker.second_max_temperature_change:.1f}K@{simulation.temperature_tracker.second_max_temperature_change_location}")
        print("="*50+"\n")

    def save_droplet_mass(self, time, droplet_params):
        if self.iteration_count % self.appearance_save_interval == 0:
            writer = csv.writer(self.mass_file)
            row = [time] + list(droplet_params.mass_i)
            writer.writerow(row)
            self.mass_file.flush()

    def save_quasi_steady_params(self, time, calc_params):
        """save the quasi-steady gas phase parameters
        
        Args:
            time: current time
            calc_params: QSParameters object
        """
        if self.quasi_steady_file is None:
            return
            
        if self.iteration_count % (self.appearance_save_interval) == 0:
        #if self.iteration_count:
            writer = csv.writer(self.quasi_steady_file)
            row = [
                time, calc_params.y_fs, calc_params.B_M, calc_params.B_T, 
                calc_params.Le, calc_params.Q_dot, calc_params.Q_gas, calc_params.Q_hv,
                calc_params.mdot, calc_params.D_ref, calc_params.dys_dr, calc_params.rho_ref
            ] + list(calc_params.E_i)  # 40-dimensional liquid phase evaporation rate
            writer.writerow(row)
            self.quasi_steady_file.flush()

    def increment_iteration(self):
        self.iteration_count += 1

    def close(self):
        self.liquid_file.close()
        self.gas_file.close()
        self.appearance_file.close()
        self.mass_file.close()
        if self.quasi_steady_file is not None:
            self.quasi_steady_file.close()

    def save_all(self, simulation):
        """
        save all the data of the current time step
        """
        self.save_liquid(simulation.runtime.current_time, simulation.liquid_array, simulation.grid)
        self.liquid_file.flush()  # ensure the data is written to the disk
        
        self.save_gas(simulation.runtime.current_time, simulation.gas_array, simulation.grid)
        self.gas_file.flush()  # ensure the data is written to the disk
        
        self.save_appearance(simulation.runtime.current_time, simulation.grid, simulation, simulation.surface)
        self.appearance_file.flush()  # ensure the data is written to the disk
        
        self.save_droplet_mass(simulation.runtime.current_time, simulation.droplet_params)
        self.mass_file.flush()  # ensure the data is written to the disk
        
        # save the quasi-steady gas phase parameters (if applicable)
        if (self.quasi_steady_file is not None and 
            hasattr(simulation, 'gas_solver') and 
            hasattr(simulation.gas_solver, 'calc_params')):
            self.save_quasi_steady_params(simulation.runtime.current_time, simulation.gas_solver.calc_params)
            self.quasi_steady_file.flush() 