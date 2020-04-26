import subprocess
import numpy as np
import re
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import minimize_scalar
from air_data import change_air_data
from qmil_design import design_opt_rpm, change_prop_area

def get_nums(dBeta, vel, thrust, prop="best_prop", motor="est_motor"):
    dBeta = str(dBeta)
    vel = str(vel)
    thrust = str(thrust)
    # Run qprop in bash with velocity, dbeta, thrust
    cmd = ['qprop', prop, motor, vel, '0', '0', dBeta, thrust]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output = process.communicate()[0]
    # Grab efficiency (18th line)
    line = str(output.splitlines()[17])
    nums = re.findall("(\d+\.)(\d+)?(E[+-]\d+)?", line)
    data = []
    for n in nums:
        try:
            data.append(float(''.join(n)))
        except:
            print("can't cast to float", n)
    try:
        effp = data[9]
        # Sanity check on efficiency
        if not (0.01 < effp < 0.99):
            data = np.zeros(19)
    # If error, condition likely invalid
    except:
        data = np.zeros(19)
    return data

def get_eff(dBeta, vel, thrust, prop="best_prop", motor="est_motor"):
    """
    Get negative of efficiency given an rpm and velocity
    """
    # Grab efficiency
    data = get_nums(dBeta, vel, thrust, prop, motor)
    eff_prop = -data[9]
    Pshaft = data[5]
    motor_loss = np.tanh(3 * Pshaft/2000)
    eff_tot = -data[14] * motor_loss
    Pelec = data[15]
    return eff_prop

def opt_dbeta(vel, thrust, prop="best_prop", motor="est_motor"):
    """
    Optimize for efficiency on variable pitch given an airspeed and required
    thrust.
    """
    res = minimize_scalar(get_eff, method='brent',
                          args=(vel, thrust, prop, motor), options={'xtol': 1e-1})
    # res = minimize_scalar(get_eff, bounds=(3.5,30), method='bounded',
    #                       args=(vel, thrust, prop, motor), options={'xatol': 1e-2})
    if res.success:
        # print(-res.fun, res.x, res.nfev)
        return res
    else:
        print("Unsuccessful optimization", res.x, res.fun)

def opt_sweep(prop="best_prop", motor="est_motor"):
    # make performance plot thrust vs v vs eta
    vels = np.arange(20, 64, 4)
    thrusts = np.arange(5, 120, 5)
    alts = np.arange(17000, 30000, 1000)
    areas = np.arange(20, 36, 2)
    dbetas = np.arange(-10, 20, 4)
    # effs = np.zeros((len(alts), len(vels), len(thrusts)))
    effs = np.zeros((len(areas), len(alts), len(vels), len(thrusts), len(dbetas)))
    ps = np.zeros((len(areas), len(alts), len(vels), len(thrusts), len(dbetas)))
    # dbetas = np.zeros((len(alts), len(vels), len(thrusts)))
    # effs_diffs = np.zeros((len(vels), len(thrusts)))
    timer = time.time()
    for i, area in enumerate(areas):
        change_prop_area(area)
        for j, h in enumerate(alts):
            change_air_data(h)
            for k, vel in enumerate(vels):
                for l, thrust in enumerate(thrusts):
                    for m, dbeta in enumerate(dbetas):
                        data = get_nums(dbeta, vel, thrust, prop, motor)
                        Pshaft = data[5]
                        eff = data[9]
                        effs[i,j,k,l,m] = eff
                        ps[i,j,k,l,m] = Pshaft

        print(np.round(i/len(areas)*100,1))
    print(time.time() - timer)
    np.savez("opt_sweep", areas=areas, hs=alts, vs=vels, ts=thrusts, dbetas=dbetas, effs=effs, ps=ps)


if __name__ == "__main__":
    prop = "best_prop"
    motor = "est_motor"
    # opt_sweep(prop, motor)

    data = get_nums(0, 10, 38)
    print(data)
    # Effect of variable pitch
    # change_air_data(0)
    # pitches = np.linspace(-15, 20, 50)
    # effs_p = np.zeros(len(pitches))
    # for i, p in enumerate(pitches):
    #     effs_p[i] = -get_eff(p, 10, 38)
    # plt.plot(pitches, effs_p)
    # plt.show()

    # Optimized effiiencies
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # X, Y = np.meshgrid(thrusts, vels)
    # surf = ax.plot_surface(X, Y, effs2, vmin=0.4, vmax=0.9, cmap=cm.coolwarm, antialiased=True, zorder = 0.5)
    # ax.set_xlabel("T (N)", fontsize=16)
    # ax.set_ylabel("V (m/s)", fontsize=16)
    # ax.set_zlabel(r"Optimized Prop $\eta$", fontsize=16)
    # ax.set_title("h = 30km", fontsize=20)
    # fig.colorbar(surf, shrink=0.8, aspect=10)

    # Unoptimized Efficiencies
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # X, Y = np.meshgrid(thrusts, vels)
    # surf = ax.plot_surface(X, Y, effs, cmap=cm.coolwarm)
    # ax.set_xlabel("Thrusts (N)")
    # ax.set_ylabel("Airspeed (m/s)")
    # ax.set_zlabel("Propeller Efficiency")
    # fig.colorbar(surf, shrink=0.5, aspect=10)

    # dBetas
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, dbetas, cmap=cm.cool)
    # ax.set_xlabel("T (N)", fontsize=16)
    # ax.set_ylabel("V (m/s)", fontsize=16)
    # ax.set_zlabel(r"Optimal $\Delta \beta$ (deg)", fontsize=16)
    # ax.set_title("Optimized Variable Pitch Angles", fontsize=18)
    # fig.colorbar(surf, shrink=0.7, aspect=10)

    # dEtas
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, effs_diffs, cmap=cm.winter)
    # ax.set_xlabel("T (N)", fontsize=16)
    # ax.set_ylabel("V (m/s)", fontsize=16)
    # ax.set_zlabel(r"$\Delta \eta$", fontsize=16)
    # ax.set_title("Increase in Prop Efficiency from Variable Pitch", fontsize=16)
    # fig.colorbar(surf, shrink=0.7, aspect=10)
    #
    # plt.show()
