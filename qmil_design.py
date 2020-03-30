import subprocess
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def get_num(line):
    """
    Regex function to find efficiency number in qmil output
    """
    nums = re.findall("\d+\.\d+", line)
    try:
        num = float(nums[0])
    except:
        # catch error of efficiency not found
        print("no number found here")
        return 0
    if len(nums) > 1 or num > 1 or num < 0:
        # check if too many values found or efficiency out of range
        print("something wrong")
        return 0
    return num

def get_eff(rpm, out_file=""):
    """
    Rewrites template qmil file and replaces desired rpm and returns
    efficiency of new designed prop
    If out_file is specified, that prop geometry is saved to the given filename
    """
    with open('template.mil', 'r') as file:
        data = file.readlines()
    data[18] = "  " + str(rpm) + " ! rpm\n"
    with open('output.mil', 'w') as file:
        file.writelines(data)

    filename = 'output.mil'

    cmd = ['qmil', filename, out_file]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output = process.communicate()[0]
    for line in output.splitlines():
        l = str(line)
        if "eff = " in l:
            eff = get_num(l)
    process.wait()
    return -eff

def plot_prop(propfile):
    f = open(propfile, "r")
    contents = f.readlines()
    prop_geom =  contents[-32:]
    P = len(prop_geom)
    r = []
    c = []
    beta = []
    R = 1.3
    for i in range(1,P):
        line = prop_geom[i].split()
        # print(line)
        r.append(float(line[0])/R)
        c.append(float(line[1])/R)
        beta.append(float(line[2]))
    f = plt.figure()
    plt.subplot(211)
    plt.plot(r,c,'b')
    plt.ylabel("c/R")
    plt.title("Propeller, R = 1.3m")
    plt.subplot(212)
    plt.plot(r, beta, 'b', label = "QMIL")
    plt.ylabel(r"$\beta$")
    plt.xlabel('r/R')

    f1 = plt.figure()
    c = np.array(c)
    plt.plot(r, c/2)
    plt.plot(r, -c/2)
    plt.xlim(0, 1)
    plt.ylim(-0.5, 0.5)
    plt.title("Propeller, R = 1.3m")
    plt.ylabel("x/R")
    plt.xlabel('y/R')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def opt_rpm():
    """
    Optimize for efficiency on rpm given an airspeed
    """
    res = minimize_scalar(get_eff, bounds=(200, 1500), method='bounded')
    if res.success:
        get_eff(res.x, "best_prop")
        print(-res.fun, res.x)
        plot_prop("best_prop")
        return res.x
    else:
        print("Unsuccessful optimization", res.x, res.fun)

opt_rpm()
# plot_prop("best_prop")
# rpms = np.arange(200.0, 1500.0, 50.0)
# etas = np.zeros(len(rpms))
#
# for i, rpm in enumerate(rpms):
#     eff = get_eff(rpm)
#     etas[i] = eff
#
# print(etas)
# best_rpm_i = np.argmax(etas)
# print(best_rpm_i, rpms[best_rpm_i], etas[best_rpm_i])
# get_eff(rpms[best_rpm_i], "best_prop")
