# prop-design
Python wrapped QMil and QProp used to design propellers for HALE UAV in 16.82 Flight Vehicle Development

## How to Use
Important functions are sectioned off into different files.

### air_data.py
This file is used to modify qcon.def which specifies air data for Qmil and Qprop and accepts an altitude when run in the terminal as an argument. Data is pulled from air_data.csv and interpolated.

### qmil_design.py
This file takes template.mil and optimizes over rpm for max efficiency to design a prop at the given conditions. The final .mil file used is saved in output.mil and the prop is saved in best_prop

### trajectory.py
This file takes in a 24 hour cycle trajectory and computes various propulsive data over the cycle. The data is pulled from time_altitude_speed.npz which contains arrays for time, altitude, velocity, and thrust

### qprop_sweep.py
This file sweeps over propulsive area, altitude, airspeed, thrust, and variable pitch angle to calculate efficiency and shaft power at each condition. This data is then saved to opt_sweep.npz

### curve_fit.py
This file takes the sweep data from qprop_sweep and attempts to fit a curve to the data.
