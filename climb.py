import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from air_data import change_air_data
from qprop_sweep import opt_dbeta, get_nums
from qmil_design import design_prop
plt.style.use('seaborn')

def follow_trajectory(ts, hs, vs, thrusts, npt=60, optimize=True, show=False,
                      prop="best_prop", save=True):
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
    prop : string, optional
        Propeller to use for analysis
    save : boolean, optional
        If true, the trajectory data will save as an npz file
    Returns
    ---------
    eff : float
        Average efficiency over 24h cycle
    """
    step = len(ts) // npt if npt <= len(ts) else 1
    result = []
    last_h = -100
    # Iterate through time steps
    for i, t in enumerate(ts[::step]):
        i *= step
        h, v, thrust = (hs[i], vs[i], thrusts[i])
        p = thrust * v
        # If alt change > 100m, recalculate air data
        # if abs(h-last_h) > 100:
        change_air_data(h)
        last_h = h
        if optimize:
            opt = opt_dbeta(v, thrust, prop=prop)
            dbeta = opt.x
            eta_opt = -opt.fun
        else:
            dbeta = 0
        data = get_nums(dbeta, v, thrust, prop=prop)
        rpm = data[1]
        Q = data[4]
        Pshaft = data[5]
        eta = data[9]
        J = data[10]
        eta_tot = data[14]
        result.append([t, h, v, p, thrust, rpm, Q, Pshaft, J, dbeta, eta, eta_tot])
    result = np.array(result)
    # print("Average Efficiency:", np.mean(result[:,10]))
    if show:
        plt.figure()
        plt.title('24 Hour Trajectory')
        plt.subplot(211)
        plt.plot(result[:,0], result[:,1]/1000, 'o-')
        plt.ylabel('Altitude [km]')
        plt.subplot(212)
        plt.plot(result[:,0], result[:,2], 'o-')
        plt.ylabel('Airspeed [m/s]')
        plt.xlabel("time [hr]")
        plt.show()
    if save:
        savefile = "climb" if optimize else "climb_unopt"
        np.savez(savefile, res=result)
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
    t, h, v, p, thrust, rpm, Q, Pshaft, J, dbeta, eta, eta_tot = result
    print("Avg RPM", np.mean(rpm), "PShaft", np.mean(Pshaft), "Q", np.mean(Q))
    print("Max RPM", max(rpm), "PShaft", max(Pshaft), "Q", max(Q))

    fig = plt.figure(figsize=(12,7.5))
    ax1 = plt.subplot(321)
    # fig.suptitle("Climb for each Propeller")
    ax1.plot(t, h/304.88, ".-", color="cornflowerblue")
    ax1.set_ylabel('Altitude [kft]', color="midnightblue")
    # ax12 = ax1.twinx()
    ax12 = plt.subplot(322)
    ax12.plot(t, rpm, ".-", color="indianred")
    ax12.set_ylabel("RPM", color="maroon")
    # ax12.grid(None)

    ax2 = plt.subplot(323)
    ax2.plot(t, thrust, ".-", color="cornflowerblue")
    ax2.set_ylabel('Thrust [N]', color="midnightblue")
    # ax22 = ax2.twinx()
    ax22 = plt.subplot(324)
    ax22.plot(t, eta, ".-", color="indianred")
    ax22.set_ylabel(r'$\eta$', color="maroon")
    # ax22.grid(None)

    ax3 = plt.subplot(325)
    ax3.plot(t, Pshaft, ".-", color="indianred")
    ax3.set_ylabel('Shaft Power [W]', color="maroon")
    # ax32 = ax3.twinx()
    ax32 = plt.subplot(326)
    ax32.plot(t, Q, ".-", color="indianred")
    ax32.set_ylabel('Required Torque [Nm]', color="maroon")
    ax3.set_xlabel("Time after Takeoff [hr]")
    ax32.set_xlabel("Time after Takeoff [hr]")
    plt.tight_layout()
    # ax32.grid(None)

    plt.figure()
    plt.plot(J, eta, ".-")
    plt.xlabel("Advance Ratio")
    plt.ylabel(r"$\eta$")
    plt.title("Advance Ratio vs Efficiency")

    plt.figure()
    plt.plot(rpm, Q, ".-")
    plt.xlabel("RPM")
    plt.ylabel("Torque Requirement [N/m]")
    plt.title("Motor Torque Requirement vs RPM for Ascent")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    num_motor = 4
    data =  np.load('climb_path.npz')
    ts = data['t'][:81]/3600
    hs = data['h'][:81]
    vs = data['v'][:81]
    thrusts = data['thrust'][:81]/num_motor

    # start = time.time()
    # eff_opt = follow_trajectory(ts, hs, vs, thrusts, npt=200, optimize=True)
    # print("Var Pitch Average Efficiency:", eff_opt)
    # plot_trajectory("climb.npz")
    # print(time.time() - start)
    eff_unopt = follow_trajectory(ts, hs, vs, thrusts, npt=200, optimize=False)
    print("Fixed Pitch Average Efficiency:", eff_unopt)
    plot_trajectory("climb_unopt.npz")
    # print(time.time() - start)
