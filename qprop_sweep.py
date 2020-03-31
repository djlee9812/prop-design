import subprocess
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import minimize_scalar

def get_nums(line):
    nums = re.findall("(\d+\.)(\d+)?(E[+-]\d+)?", line)
    result = []
    for n in nums:
        try:
            result.append(float(''.join(n)))
        except:
            print("can't cast to float", n)
    return result

def get_eff(dBeta, vel, thrust):
    """
    Get negative of efficiency given an rpm and velocity
    """
    dBeta = str(dBeta)
    vel = str(vel)
    thrust = str(thrust)
    cmd = ['qprop', prop, motor, vel, '0', '0', dBeta, thrust]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output = process.communicate()[0]
    data = get_nums(str(output.splitlines()[17]))
    try:
        effp = data[9]
        rpm = data[1]
    except:
        effp = 0
        rpm = 0
    if not (0.01 < effp < 0.99):
        effp = 0
        rpm = 0
        # print('(vel, rpm) combo invalid', vel, rpm)
    process.wait()
    # print(vel, thrust, rpm, dBeta)
    return -effp

def opt_dbeta(vel, thrust):
    """
    Optimize for efficiency on variable pitch given an airspeed and required
    thrust.
    """
    res = minimize_scalar(get_eff, bounds=(-30, 30), method='bounded',
                          args=(vel, thrust), options={'xatol': 1e-1})
    if res.success:
        # print(-res.fun, res.x)
        return res
    else:
        print("Unsuccessful optimization", res.x, res.fun)

prop = "best_prop"
motor = "est_motor"

# eff = get_eff("30.0", "200.0")

# best_pitch = opt_dbeta(20, 400)
# print(best_pitch.x)

# pitches = np.linspace(-30, 30, 50)
# effs_p = np.zeros(len(pitches))
# for i, p in enumerate(pitches):
#     effs_p[i] = -get_eff(p, 40, 800)
# plt.plot(pitches, effs_p)
# plt.show()

# make performance plot thrust vs v vs eta
vels = np.arange(20, 64, 4)
thrusts = np.arange(5, 120, 5)
effs = np.zeros((len(vels), len(thrusts)))
effs2 = np.zeros((len(vels), len(thrusts)))
dbetas = np.zeros((len(vels), len(thrusts)))
effs_diffs = np.zeros((len(vels), len(thrusts)))
for i, vel in enumerate(vels):
    for j, thrust in enumerate(thrusts):
        opt = opt_dbeta(vel, thrust)
        effs[i,j] = -get_eff(0, vel, thrust)
        effs2[i,j] = -opt.fun
        dbetas[i, j] = opt.x
        effs_diffs[i, j] = effs2[i,j] - effs[i,j]
    if i % 3 == 0:
        print(np.round(i/len(vels)*100,1))

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(thrusts, vels)
surf = ax.plot_surface(X, Y, effs2, vmin=0.4, vmax=0.9, cmap=cm.coolwarm, antialiased=True, zorder = 0.5)
ax.set_xlabel("T (N)", fontsize=16)
ax.set_ylabel("V (m/s)", fontsize=16)
ax.set_zlabel(r"Optimized Prop $\eta$", fontsize=16)
ax.set_title("h = 30km", fontsize=20)
fig.colorbar(surf, shrink=0.8, aspect=10)
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X, Y = np.meshgrid(thrusts, vels)
# surf = ax.plot_surface(X, Y, effs, cmap=cm.coolwarm)
# ax.set_xlabel("Thrusts (N)")
# ax.set_ylabel("Airspeed (m/s)")
# ax.set_zlabel("Propeller Efficiency")
# fig.colorbar(surf, shrink=0.5, aspect=10)


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, dbetas, cmap=cm.cool)
ax.set_xlabel("T (N)", fontsize=16)
ax.set_ylabel("V (m/s)", fontsize=16)
ax.set_zlabel(r"Optimal $\Delta \beta$ (deg)", fontsize=16)
ax.set_title("Optimized Variable Pitch Angles", fontsize=18)
fig.colorbar(surf, shrink=0.7, aspect=10)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, effs_diffs, cmap=cm.winter)
ax.set_xlabel("T (N)", fontsize=16)
ax.set_ylabel("V (m/s)", fontsize=16)
ax.set_zlabel(r"$\Delta \eta$", fontsize=16)
ax.set_title("Increase in Prop Efficiency from Variable Pitch", fontsize=16)
fig.colorbar(surf, shrink=0.7, aspect=10)

plt.show()
