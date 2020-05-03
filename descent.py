import subprocess
import numpy as np
import re
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from air_data import change_air_data

def get_qprop_cl(output):
    prop_data = output[20:]
    index = int(len(prop_data) * 4 / 5)
    nums = re.findall("(-?)(\d+)(\.)?(\d+)?(E[+-]\d+)?", str(prop_data[index]))
    data = []
    for n in nums:
        try:
            data.append(float(''.join(n)))
        except:
            print("can't cast to float", n)
    return data[3], data[4]

def run_qprop(vel, dbeta=0, prop="best_prop", motor="est_motor"):
    """
    Function to run qprop for optimizer model data collection
    """
    vel = str(vel)
    dbeta = str(dbeta)
    # Run qprop in bash with velocity, rpm
    cmd = ['qprop', prop, motor, vel, '0', '0', dbeta, '0', '0.1']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output = process.communicate()[0]
    cl, cd = get_qprop_cl(output.splitlines())
    # Grab efficiency (18th line)
    line = str(output.splitlines()[17])
    nums = re.findall("(-?)(\d+\.)(\d+)?(E[+-]\d+)?", line)
    data = []
    for n in nums:
        try:
            data.append(float(''.join(n)))
        except:
            print("can't cast to float", n)
    return data, cl, cd

def windmill(alt_min, alt_max, dbeta):
    q = 45
    num_alts = 50
    alts = np.linspace(alt_min, alt_max, num_alts)
    result = np.array([np.zeros(12) for i in range(num_alts)])
    result[:,0] = alts
    for i, alt in enumerate(alts):
        rho, mu, a = change_air_data(alt)
        v = np.round((q * 2 / rho) ** 0.5, 2)
        result[i,1] = rho
        data, cl, cd = run_qprop(v, dbeta)
        result[i,2] = cl
        result[i,3] = cd
        result[i,4:] = data[:8]
    return result

def plot_descent(result):
    plt.figure()
    plt.plot(result[:,0]/1e3, result[:,7])
    plt.xlabel("Altitude [km]")
    plt.ylabel("Thrust [N]")
    plt.show()


if __name__ == "__main__":
    avg_lift = 390 * 9.8
    num_motor = 4
    # run_qprop(32.01, -10)
    result0 = windmill(0, 20000, 0)
    result5 = windmill(0, 20000, -5)
    result10 = windmill(0, 20000, -10)
    # result15 = windmill(0, 20000, -15)
    plt.figure()
    plt.plot(result0[:,0]/1e3, -result0[:,7]/avg_lift*num_motor, label=r"$d\beta = 0\degree$")
    plt.plot(result5[:,0]/1e3, -result5[:,7]/avg_lift*num_motor, label=r"$d\beta = -5\degree$")
    plt.plot(result10[:,0]/1e3, -result10[:,7]/avg_lift*num_motor, label=r"$d\beta = -10\degree$")
    # plt.plot(result15[:,0]/1e3, -result15[:,7]/avg_lift*num_motor, label=r"$d\beta = -15\degree$")
    plt.xlabel("Altitude [km]")
    plt.ylabel("Windmill Drag / Average Lift")
    plt.title("Total Windmill Drag for Descent")
    plt.legend()
    plt.figure()
    plt.plot(result0[:,0]/1e3, result0[:,2], label=r"$d\beta = 0\degree$")
    plt.plot(result5[:,0]/1e3, result5[:,2], label=r"$d\beta = -5\degree$")
    plt.plot(result10[:,0]/1e3, result10[:,2], label=r"$d\beta = -10\degree$")
    # plt.plot(result15[:,0]/1e3, result15[:,2], label=r"$d\beta = -15\degree$")
    plt.xlabel("Altitude [km]")
    plt.ylabel("Blade CL")
    plt.title("Blade Windmill cl for Descent")
    plt.legend()
    plt.figure()
    plt.plot(result0[:,0]/1e3, result0[:,3], label=r"$d\beta = 0\degree$")
    plt.plot(result5[:,0]/1e3, result5[:,3], label=r"$d\beta = -5\degree$")
    plt.plot(result10[:,0]/1e3, result10[:,3], label=r"$d\beta = -10\degree$")
    # plt.plot(result15[:,0]/1e3, result15[:,3], label=r"$d\beta = -15\degree$")
    plt.xlabel("Altitude [km]")
    plt.ylabel("Blade CD")
    plt.title("Blade Windmill cd for Descent")
    plt.legend()
    plt.show()
