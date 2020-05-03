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
    last_h = 0
    # Iterate through time steps
    for i, t in enumerate(ts[::step]):
        i *= step
        h, v, thrust = (hs[i], vs[i], thrusts[i])
        p = thrust * v
        # If alt change > 100m, recalculate air data
        if abs(h-last_h) > 100:
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
        result.append([t, h, v, p, thrust, rpm, Q, Pshaft, J, dbeta, eta])
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
        savefile = "cycle" if optimize else "cycle_unopt"
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
    t, h, v, p, thrust, rpm, Q, Pshaft, J, dbeta, eta = result
    night = int( len(t) * 7.5 // 24 )
    day = len(t) * 18 // 24
    res_day = np.concatenate((result[:,:night], result[:,day:]), axis=1)
    res_night = result[:,night:day]
    t_d, h_d, v_d, p_d, thrust_d, rpm_d, Q_d, Pshaft_d, J_d, dbeta_d, eta_d = res_day
    t_n, h_n, v_n, p_n, thrust_n, rpm_n, Q_n, Pshaft_n, J_n, dbeta_n, eta_n = res_night

    print("Avg RPM", rpm_n[0], "PShaft", Pshaft_n[0], "Q", Q_n[0])
    print("Max RPM", max(rpm_d), "PShaft", max(Pshaft_d), "Q", max(Q_d))

    fig = plt.figure(figsize=(7,7.5))
    ax1 = plt.subplot(311)
    ax1.set_title("24 Hour Cycle for each Propeller")
    ax1.plot(t_d, h_d/304.88, ".-", label="Day", color="cornflowerblue")
    ax1.plot(t_n, h_n/304.88, ".-", label="Night", color="midnightblue")
    ax1.set_ylabel('Altitude [kft]', color="midnightblue")
    ax12 = ax1.twinx()
    ax12.plot(t_d, rpm_d, ".-", label="Day", color="indianred")
    ax12.plot(t_n, rpm_n, ".-", label="Night", color="maroon")
    ax12.set_ylabel("RPM", color="maroon")
    ax12.grid(None)
    # ax12.set_yticks(np.linspace(ax12.get_yticks()[0], ax12.get_yticks()[-1], len(ax1.get_yticks())))

    ax2 = plt.subplot(312)
    ax2.plot(t_d, thrust_d, ".-", label="Day", color="cornflowerblue")
    ax2.plot(t_n, thrust_n, ".-", label="Night", color="midnightblue")
    ax2.set_ylabel('Thrust [N]', color="midnightblue")
    ax22 = ax2.twinx()
    ax22.plot(t_d, eta_d, ".-", label="Day", color="indianred")
    ax22.plot(t_n, eta_n, ".-", label="Night", color="maroon")
    ax22.set_ylabel(r'$\eta$', color="maroon")
    ax22.grid(None)

    ax3 = plt.subplot(313)
    # plt.plot(t, dbeta)
    # plt.ylabel(r'$d\beta$')
    ax3.plot(t_d, Pshaft_d, ".-", label="Day", color="cornflowerblue")
    ax3.plot(t_n, Pshaft_n, ".-", label="Night", color="midnightblue")
    ax3.set_ylabel('Shaft Power [W]', color="midnightblue")
    ax32 = ax3.twinx()
    ax32.plot(t_d, Q_d, ".-", label="Day", color="indianred")
    ax32.plot(t_n, Q_n, ".-", label="Night", color="maroon")
    ax32.set_ylabel('Required Torque [Nm]', color="maroon")
    ax3.set_xlabel("Time from Solar Noon [hr]")
    ax32.grid(None)

    plt.figure()
    plt.plot(J, eta, ".-")
    plt.xlabel("Advance Ratio")
    plt.ylabel(r"$\eta$")
    plt.title("Advance Ratio vs Efficiency")

    plt.figure()
    plt.plot(rpm, Q, ".-")
    plt.xlabel("RPM")
    plt.ylabel("Torque Requirement [N/m]")
    plt.title("Motor Torque Requirement vs RPM")
    # plt.show()

def plot_motor_eff(Kv, R, I0):
    """
    Calculate and plot motor efficiency over various torque and RPM ranges.
    The RPM/Omega used in the calculations and Kv should be in units of
    rad/s instead of rpm.
    """
    Qs = np.linspace(5, 15, 50)
    RPMs = np.linspace(200, 1800, 60)*np.pi/30
    mot_effs = np.zeros((50, 60))
    for i, Q in enumerate(Qs):
        for j, RPM in enumerate(RPMs):
            # I = Q*Kv + I0
            # V = RPM * Kv + I*R
            # mot_effs[i, j] = (1 - I*R/V) * (1 - I0/I)
            mot_effs[i,j] = (Kv*RPM/(Kv*RPM+Q*R*Kv+I0*R))*(Q*Kv/(Q*Kv+I0))
    # return np.mean(mot_effs)
    print(np.mean(mot_effs))
    plt.figure()
    X, Y = np.meshgrid(RPMs*30/np.pi, Qs)
    plt.contourf(X, Y, mot_effs)
    plt.xlabel("RPMs")
    plt.ylabel("Torques")
    plt.title("Motor Efficiencies over Operating Range")
    plt.colorbar()
    plt.show()


def rpm_trade(rpms, ts, hs, vs, thrusts, optimize):
    """
    Trade study of design RPM vs average efficiency over cycle. Given a range
    of rpms, redesign "test_prop" to be optimized at the given rpm and use
    the output propeller to get average efficiency over cycle
    """
    effs = np.zeros(len(rpms))
    for i, rpm in enumerate(rpms):
        print("Design rpm:", rpm, "Design eff:", -design_prop(rpm, "test_prop"))
        effs[i] = follow_trajectory(ts, hs, vs, thrusts, npt=200,
                                    optimize=optimize, show=False,
                                    prop="test_prop")
    # plt.figure()
    # plt.plot(rpms, effs)
    # opt_title = "with Variable Pitch" if optimize else ""
    # plt.title("Design RPM vs Average Efficiency " + opt_title)
    # plt.xlabel("QMil Design RPM")
    # plt.ylabel(r"Average $\eta$ over 24 Hour Cycle")
    # plt.show()
    return effs

if __name__ == "__main__":
    num_motor = 4
    data =  np.load('time_altitude_airspeed.npz')
    ts = data['t']/3600
    hs = data['h']
    vs = data['v']
    thrusts = data['thrust']/num_motor
    # plt.plot(ts, hs)
    # plt.show()

    # plot_motor_eff(8*np.pi/30, 0.3, 1.1)
    # Kvs = np.linspace(3, 10, 10)
    # I0s = np.linspace(1, 5, 15)
    # effs = np.zeros((10, 15))
    # for i, Kv in enumerate(Kvs):
    #     for j, I0 in enumerate(I0s):
    #         effs[i,j] = plot_motor_eff(Kv*np.pi/30, 1.1, I0)
    # plt.figure()
    # X, Y = np.meshgrid(I0s, Kvs)
    # plt.contourf(X, Y, effs)
    # plt.title("Average Efficiencies for Various Motors")
    # plt.xlabel(r"$I_0$")
    # plt.ylabel("Kv")
    # plt.colorbar()
    # plt.show()

    # rpms = np.arange(800, 1700, 50)
    # opt_eff = rpm_trade(rpms, ts, hs, vs, thrusts, True)
    # unopt_eff = rpm_trade(rpms, ts, hs, vs, thrusts, False)
    # np.savez("rpm_trade.npz", opt_eff, unopt_eff)
    # plt.figure()
    # plt.plot(rpms, opt_eff, label="Variable Pitch")
    # plt.plot(rpms, unopt_eff, label="Fixed Pitch")
    # plt.title("Design RPM vs Average Efficiency")
    # plt.xlabel("QMil Design RPM")
    # plt.ylabel(r"Average $\eta$ over 24 Hour Cycle")
    # plt.legend()
    # plt.show()

    # start = time.time()
    # eff_opt = follow_trajectory(ts, hs, vs, thrusts, npt=200, optimize=True)
    # print("Average Efficiency Var Pitch:", eff_opt)
    # plot_trajectory("cycle.npz")
    # print(time.time() - start)
    eff_unopt = follow_trajectory(ts, hs, vs, thrusts, npt=200, optimize=False)
    print("Average Efficiency Fixed Pitch:", eff_unopt)
    plot_trajectory("cycle_unopt.npz")
    plt.show()
    # print(time.time() - start)


    # result = np.load("cycle.npz")["res"].T
    # result2 = np.load("cycle_unopt.npz")["res"].T
    # t, h, v, p, thrust, rpm, Q, Pshaft, J, dbeta, eta = result
    # plt.figure()
    # plt.plot(t, rpm, label="Variable Pitch")
    # plt.ylabel('RPM')
    # result = np.load("cycle_unopt.npz")["res"].T
    # t, h, v, p, thrust, rpm, Q, Pshaft, J, dbeta, eta = result
    # plt.plot(t, rpm, label="Fixed Pitch")
    # plt.xlabel("Time [hr]")
    # plt.title("RPM over cycle")
    # plt.legend()
    # plt.show()
    #
