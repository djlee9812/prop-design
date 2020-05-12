# prop-design
Python wrapped QMil and QProp used to design propellers for HALE UAV in 16.82 Flight Vehicle Development for the Solar HALE UAV, Dawn.

## How to Use
Important functions for prop design are sectioned off into different files.

### air_data.py
This file is used to modify qcon.def which specifies air data for Qmil and Qprop and accepts an altitude when run in the terminal as an argument. There are two options for data -- to pull from air_data.csv which is more accurate but slower, or to use aerosandbox curve fit atmospheric model which is faster. The latter is the default.

### qmil_design.py
This file takes template.mil and optimizes over rpm for max efficiency to design a prop at the given conditions. The final .mil file used is saved in output.mil and the prop is saved in best_prop. Plots propeller geometry

### trajectory.py
This file takes in a 24 hour cycle trajectory and computes various propulsive data over the cycle. The data is pulled from time_altitude_speed.npz which contains arrays for time, altitude, velocity, and thrust

### climb.py
This file works similar to trajectory.py. It requires a climb_path.npz input and runs QProp for the climb trajectory.

### descent.py
The descend() function calculates a descent trajectory with windmilling props. See docstring and final memo for detailed information.
