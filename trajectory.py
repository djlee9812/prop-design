import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from air_data import change_air_data
from qprop_sweep import opt_dbeta, get_nums
from qmil_design import design_prop

def follow_trajectory(ts, hs, vs, thrusts, npt=24, optimize=True, show=True):
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
    show : boolean, optional
        If checked, will show discretized altitude and airspeed over cycle
    """
    step = len(ts) // npt if npt <= len(ts) else 1
    result = []
    last_h = 0
    for i, t in enumerate(ts[::step]):
        i *= step
        h, v, thrust = (hs[i], vs[i], thrusts[i])
        p = thrust * v
        if abs(h-last_h) > 100:
            change_air_data(h)
            last_h = h
        opt = opt_dbeta(v, thrust)
        dbeta = opt.x if optimize else 0
        eta_opt = -opt.fun
        data = get_nums(dbeta, v, thrust)
        rpm = data[1]
        Q = data[4]
        Pshaft = data[5]
        eta = data[9]
        J = data[10]
        result.append([t, h, v, p, thrust, rpm, Q, Pshaft, J, dbeta, eta])
    result = np.array(result)
    print("Average Efficiency:", np.mean(result[:,10]))
    if show:
        plt.figure()
        plt.title('24 Hour Trajectory')
        plt.subplot(211)
        plt.plot(result[:,0], result[:,1]/1000, 'o')
        plt.ylabel('Altitude [km]')
        plt.subplot(212)
        plt.plot(result[:,0], result[:,2], 'o')
        plt.ylabel('Airspeed [m/s]')
        plt.xlabel("time [hr]")
        plt.show()
    np.savez("cycle", res=result)
    return np.mean(result[:,10])

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
    plt.figure(figsize=(6,7))
    plt.title("24 Hour Cycle")
    plt.subplot(411)
    plt.plot(t, h/1000)
    plt.ylabel('Altitude [km]')
    plt.subplot(412)
    plt.plot(t, rpm)
    plt.ylabel('RPM')
    plt.subplot(413)
    # plt.plot(t, dbeta)
    # plt.ylabel(r'$d\beta$')
    plt.plot(t, Pshaft)
    plt.ylabel('Shaft Power [N]')
    plt.subplot(414)
    # plt.plot(t, eta)
    # plt.ylabel(r'$\eta$')
    plt.plot(t, Q)
    plt.ylabel('Torque [N/m]')
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

def rpm_trade(ts, hs, vs, thrusts, optimize):
    """
    Trade study of design RPM vs average efficiency over cycle. Given a range
    of rpms, redesign "test_prop" to be optimized at the given rpm and use
    the output propeller to get average efficiency over cycle
    """
    rpms = np.arange(800, 1500, 50)
    effs = np.zeros(len(rpms))
    for i, rpm in enumerate(rpms):
        print("Design rpm:", rpm, "Design eff:", -design_prop(rpm, "test_prop"))
        effs[i] = follow_trajectory(ts, hs, vs, thrusts, npt=200,
                                    optimize=optimize, show=False)
    plt.figure()
    plt.plot(rpms, effs)
    plt.title("Design RPM vs Average Efficiency over 24 Hour Cycle Trade Study")
    plt.xlabel("RPM")
    plt.ylabel(r"$\eta$")
    plt.show()

if __name__ == "__main__":
    data =  np.load('time_altitude_airspeed.npz')
    ts = data['t']/3600
    hs = data['h']
    vs = data['v']
    thrusts = data['thrust']
    rpm_trade(ts, hs, vs, thrusts, True)

    # follow_trajectory(ts, hs, vs, thrusts, npt=200, optimize=True)
    # plot_trajectory()

    #
