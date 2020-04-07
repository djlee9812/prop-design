import subprocess
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from air_data import change_air_data

R = 1.2
NUM_MOTOR = 6
HUB_R = .15

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

def design_prop(rpm, out_file=""):
    """
    Rewrites template.qmil file to change desired rpm and returns
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

def change_prop_area(area):
    """
    Change tip_radius in template.qmil using given total propulsive area and
    scaled according to number of motors and hub diameter
    """
    tip_r = np.round(np.sqrt((area/NUM_MOTOR)/np.pi + HUB_R**2), 3)
    with open('template.mil', 'r') as file:
        data = file.readlines()
    data[16] = "  " + str(tip_r) + "   ! tip radius(m)\n"
    with open('template.mil', 'w') as file:
        file.writelines(data)
    design_opt_rpm()

def plot_prop(propfile):
    f = open(propfile, "r")
    contents = f.readlines()
    prop_geom =  contents[-32:]
    P = len(prop_geom)
    r = []
    c = []
    beta = []
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
    plt.title(r"Propeller Chord and $\beta$, R = " + str(R) + "m")
    plt.subplot(212)
    plt.plot(r, beta, 'b', label = "QMIL")
    plt.ylabel(r"$\beta$")
    plt.xlabel('r/R')

    hub_x = np.arange(0, .162, 0.002)
    hub_y = np.sqrt(0.16**2 - hub_x**2)
    hub_x = np.concatenate([hub_x, hub_x[::-1]])
    hub_y = np.concatenate([hub_y, -hub_y[::-1]])
    fig, ax = plt.subplots()
    c = np.array(c)
    prop_r = np.concatenate([r, r[::-1]])
    prop_c = np.concatenate([c/2, -c[::-1]/2])
    plt.plot(prop_r, prop_c, 'b', label="Flattened propeller")
    plt.plot(hub_x, hub_y, 'r', label="Propeller hub")
    # circle = plt.Circle((0, 0), 0.15, color='r')
    plt.xlim(0, 1)
    plt.ylim(-0.5, 0.5)
    plt.title("Propeller, R = " + str(R) + "m")
    plt.ylabel("x/R")
    plt.xlabel('y/R')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    # ax.add_artist(circle)
    plt.show()

def design_opt_rpm(h=22000):
    """
    Design a propeller by optimizing for efficiency on rpm for flight conditions
    given in template.mil and optional argument altitude
    """
    change_air_data(h)
    res = minimize_scalar(design_prop, bounds=(400, 1600), method='bounded')
    if res.success:
        design_prop(res.x, "best_prop")
        # print(-res.fun, res.x)
        # plot_prop("best_prop")
        return res.x
    else:
        print("Unsuccessful optimization", res.x, res.fun)

if __name__ == "__main__":
    design_opt_rpm(22000)
    # change_prop_area(27)

    # print(-design_prop(1200, "test_prop"))
    # plot_prop('test_prop')

    # rpms = np.arange(600.0, 2500.0, 100.0)
    # etas = np.zeros(len(rpms))
    #
    # for i, rpm in enumerate(rpms):
    #     eff = design_prop(rpm)
    #     etas[i] = eff
    # plt.plot(rpms, -etas)
    # plt.xlabel("rpm")
    # plt.ylabel(r"$\eta$")
    # plt.show()

    # best_rpm_i = np.argmax(etas)
    # print(best_rpm_i, rpms[best_rpm_i], etas[best_rpm_i])
    # design_prop(rpms[best_rpm_i], "best_prop")
