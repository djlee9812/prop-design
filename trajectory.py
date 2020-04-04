import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from air_data import change_air_data
from qprop_sweep import opt_dbeta, get_nums

def follow_trajectory(ts, hs, vs, thrusts, npt=24, unopt=False):
    """
    Calculate propulsion parameters over a cycle trajectory and saves data
    to "cycle.npz" using QProp
    Requires time, altitude, airspeed, and thrust over the 24 hour trajectory

    Parameters
    ----------
    ts : array-like
        Points in time
    hs : array-like
        Altitudes at given time points
    vs : array-like
        Airspeeds at given time poitns
    thrusts : array-like
        Required thrust at given time points
    npt : int, optional
        Number of time samples to evaluate/optimize at
    unopt : boolean, optional
        If checked, variable pitch is not allowed
    """
    step = len(ts) // npt if npt <= len(ts) else 1
    result = []
    for i, t in enumerate(ts[::step]):
        i *= step
        h, v, thrust = (hs[i], vs[i], thrusts[i])
        p = thrust * v
        change_air_data(h)
        opt = opt_dbeta(v, thrust)
        dbeta = 0 if unopt else opt.x
        eta_opt = -opt.fun
        data = get_nums(dbeta, v, thrust)
        rpm = data[1]
        Q = data[4]
        Pshaft = data[5]
        eta = data[9]
        J = data[10]
        result.append([t, h, v, p, thrust, rpm, Q, Pshaft, J, dbeta, eta])
    result = np.array(result)
    plt.figure()
    plt.title('24 Hour Trajectory')
    plt.subplot(211)
    plt.plot(result[:,0], result[:,1]/1000)
    plt.ylabel('Altitude [km]')
    plt.subplot(212)
    plt.plot(result[:,0], result[:,2])
    plt.ylabel('Airspeed [m/s]')
    plt.xlabel("time [hr]")
    plt.show()
    np.savez("cycle", res=result)
    print("Average Efficiency:", np.mean(result[:,10]))

def plot_trajectory(data_file="cycle.npz"):
    """
    Plot various propulsion parameters from cycle.npz

    Parameters
    ----------
    data_file : str, optional
        Name/Address of file containing plot data (from follow_trajectory())
    """
    result = np.load(data_file)["res"].T
    t, h, v, p, thrust, rpm, Q, Pshaft, J, dbeta, eta = result
    plt.figure(figsize=(6,8))
    plt.title("24 Hour Cycle")
    plt.subplot(411)
    plt.plot(t, h/1000)
    plt.ylabel('Altitude [km]')
    plt.subplot(412)
    plt.plot(t, rpm)
    plt.ylabel('RPM')
    plt.subplot(413)
    plt.plot(t, dbeta)
    plt.ylabel(r'$d\beta$')
    plt.subplot(414)
    plt.plot(t, eta)
    plt.ylabel(r'$\eta$')
    plt.xlabel("Time [hr]")

    plt.figure()
    plt.plot(J, eta)
    plt.xlabel("Advance Ratio")
    plt.ylabel(r"$\eta$")
    plt.figure()
    plt.plot(rpm, Q)
    plt.xlabel("RPM")
    plt.ylabel("Torque [N/m]")
    plt.show()

data =  np.load('time_altitude_airspeed.npz')
ts = data['t']/3600
hs = data['h']
vs = data['v']
thrusts = data['thrust']
follow_trajectory(ts, hs, vs, thrusts, 200)

plot_trajectory()
