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
    except:
        effp = 0
    if not (0.01 < effp < 0.99):
        effp = 0
        # print('(vel, rpm) combo invalid', vel, rpm)
    process.wait()
    return -effp

def opt_dbeta(vel, thrust):
    """
    Optimize for efficiency on variable pitch given an airspeed and required
    thrust.
    """
    res = minimize_scalar(get_eff, bounds=(-30, 30), method='bounded', args=(vel, thrust))
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

pitches = np.linspace(-30, 30, 50)
effs_p = np.zeros(len(pitches))
for i, p in enumerate(pitches):
    effs_p[i] = -get_eff(p, 40, 800)
plt.plot(pitches, effs_p)
plt.show()
# make performance plot thrust vs v vs eta
# vels = np.arange(2, 40, 4)
# thrusts = np.arange(50, 800, 50)
# effs = np.zeros((len(vels), len(thrusts)))
# effs2 = np.zeros((len(vels), len(thrusts)))
# for i, vel in enumerate(vels):
#     for j, thrust in enumerate(thrusts):
#         effs[i,j] = -get_eff(0, vel, thrust)
#         effs2[i,j] = -opt_dbeta(vel, thrust).fun
#     if i % 3 == 0:
#         print(np.round(i/len(vels)*100,1))
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X, Y = np.meshgrid(thrusts, vels)
# surf = ax.plot_surface(X, Y, effs2, cmap=cm.coolwarm)
# ax.set_xlabel("Thrusts (N)")
# ax.set_ylabel("Airspeed (m/s)")
# ax.set_zlabel("Optimized Propeller Efficiency")
# fig.colorbar(surf, shrink=0.5, aspect=10)
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X, Y = np.meshgrid(thrusts, vels)
# surf = ax.plot_surface(X, Y, effs, cmap=cm.coolwarm)
# ax.set_xlabel("Thrusts (N)")
# ax.set_ylabel("Airspeed (m/s)")
# ax.set_zlabel("Propeller Efficiency")
# fig.colorbar(surf, shrink=0.5, aspect=10)
# plt.show()
