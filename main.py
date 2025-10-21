"""
Main entry point for droplet combustion simulation

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.
"""

import numpy as np
from src.simulation import Simulation, SimulationParameters
import os
import time
import sys
from datetime import datetime
from src.core.logger import TeeLogger

# !important: set the OPENBLAS_NUM_THREADS to 1 will improve the performance of the code, if you use other blas library, you would better also to set the number of threads to 1
os.environ['OPENBLAS_NUM_THREADS'] = '1'


#* about the modification of liquid phase composition 
# liquid phase composition is set to 40 species, including straight-chain alkanes, branched alkanes, cycloalkanes, and aromatic hydrocarbons from C7 to C16. Currently
# some parts of the code use hard-coded values for the liquid phase composition. If you need to modify the number of components, you can search for 40 globally. 
# If you only need to modify the composition, you need to modify the LIQUID_SPECIES_NAMES dictionary and GAS_liquid_dict dictionary (the correspondence between liquid and gas phases) in src/utils/mapping_utils.py, and the liquid phase physical properties and liquid phase physical property calculation methods in src/solution/liquid_para.py.
#* about the modification of gas phase composition
# gas phase composition information is in the mech/*.yaml mechanism file
# it is important to note that the order of the common components in the liquid and gas phases should be the same and N2 must be the first component(for normalization).
# !important: it is necessary to check the consistency of the thermodynamic mechanism file using cantera.

def main():
    case_name = "case name"  # case name
    fuel_composition = np.zeros(40)  # create an array of 40 components
    fuel_composition[0] = 0.5  # set the molar fraction, n-hepante for example
    fuel_composition[11] = 0.5  # set the molar fraction, n-hepante for example
    # create the result directory and log file
    result_dir = os.path.join("result", case_name)
    os.makedirs(result_dir, exist_ok=True)
    log_filename = os.path.join(result_dir, f"{case_name}.log")
    
    # set the log recording
    logger = TeeLogger(log_filename)
    sys.stdout = logger
    
    try:
        # record the start time and basic information
        start_datetime = datetime.now()
        print(f"=== Droplet Combustion Simulation Log ===")
        print(f"Case Name: {case_name}")
        print(f"Start Time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log File: {log_filename}")
        print("="*50)
        
        params = SimulationParameters(
            case_name=case_name,
            fuel_composition=fuel_composition,
            droplet_radius=7.0e-4,  # droplet radius, unit[m]
            boundary_temperature=700.0,  # boundary temperature, unit[K]
            initial_pressure=1.0e6,  # initial pressure, unit[Pa]
            initial_temperature=300.0,  # initial temperature, unit[K]
            liquid_cell_count=40,  # liquid phase grid count
            gas_cell_count=200,  # gas phase grid count
            initital_time_step=1E-4,  # time step, unit[s]
            mechanism_file='Mech/zhang.yaml',  # gas phase mechanism file
            liquid_solver_type='FTCFD',  # liquid phase solver type: ITCID、FTCID、ITCFD、FTCFD
            gas_solver_type='react',  # gas phase solver type: evap、react、Quasi_Steady
            flag_fast_solver_for_transient=False # flag for fast solver for transient
        )

        # create simulation instance
        simulation = Simulation(params)

        # initialize simulation
        simulation.initialize()

        # start simulation
        start_time = time.time()
        simulation.run()
        end_time = time.time()
        total_time = end_time - start_time

        # end output
        end_datetime = datetime.now()
        print("="*50)
        print(f"Simulation completed successfully!")
        print(f"End Time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Simulation Time: {total_time:.2f} seconds")
        print(f"Total Wall Clock Time: {(end_datetime - start_datetime).total_seconds():.2f} seconds")
        print("="*50)
        
    except Exception as e:
        # record the error information
        print(f"ERROR: Simulation failed with exception: {str(e)}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        raise
    finally:
        # restore the standard output and close the log file
        sys.stdout = logger.terminal
        logger.close()

if __name__ == "__main__":
    main()