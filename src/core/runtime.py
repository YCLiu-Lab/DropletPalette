"""
runtime module

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.

This module provides the basic control functions of the simulation runtime, including time stepping, running state control, etc.

main functions:
1. time control
   - set and update the time step
   - track the current simulation time
   - control the simulation end time

2. running state management
   - control the start and stop of the simulation
   - provide the running state query interface
   - support manual stop of the simulation

3. time stepping
   - provide the time stepping function
   - ensure the accuracy of the time stepping
   - maintain the temporal continuity of the simulation
"""

class Runtime:
    def __init__(self, time_step: float = 1e-4, end_time: float = 200.0):
        """
        initialize the runtime class
        
        Args:
            time_step: time step, default 0.1ms
            end_time: end time, default 200s
        """
        self.time_step = time_step
        self.end_time = end_time
        self.current_time = 0.0
        self.running = True
        
    def is_running(self) -> bool:
        """
        check if the simulation is still running
        
        Returns:
            bool: if the current time is less than the total time and running is True, return True
        """
        return self.current_time < self.end_time and self.running
        
    def stop(self):
        """
        manually stop the simulation
        """
        self.running = False
        
    def advance(self):
        """
        advance the current time by one time step
        """
        self.current_time += self.time_step 