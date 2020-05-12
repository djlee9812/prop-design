import subprocess
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from air_data import change_air_data

R = 1.10
NUM_MOTOR = 4
HUB_R = .12

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

def design_prop(rpm, out_file="temp_prop", traj_eval=False, opt=False):
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
    if traj_eval:
        from trajectory import follow_trajectory
        traj_data =  np.load('time_altitude_airspeed.npz')
        ts = traj_data['t']/3600
        hs = traj_data['h']
        vs = traj_data['v']
        thrusts = traj_data['thrust']/NUM_MOTOR
        eff = follow_trajectory(ts, hs, vs, thrusts, optimize=opt,
                                prop="temp_prop", save=False)
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
    """
    Plot the propeller r/c and beta/c distribution and the actual geometry
    :params propfile: Filename or path to file containing propeller geometry
    """
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
    plt.axhline(y=np.mean(c), linestyle=":", color="gray")
    plt.annotate("Average c/R", xy=(0.9,np.mean(c)+.01))
    plt.ylabel("c/R", fontsize="16")
    plt.title(r"Propeller Chord and $\beta$, R = " + str(R) + "m", fontsize="18")
    plt.subplot(212)
    plt.plot(r, beta, 'b', label = "QMIL")
    plt.ylabel(r"$\beta$", fontsize="16")
    plt.xlabel('r/R', fontsize="16")

    radius = HUB_R / R
    hub_x = np.arange(0, radius, 0.002)
    hub_y = np.sqrt(radius**2 - hub_x**2)
    hub_x = np.concatenate([hub_x, hub_x[::-1]])
    hub_y = np.concatenate([hub_y, -hub_y[::-1]])
    fig, ax = plt.subplots()
    c = np.array(c)
    prop_r = np.concatenate([r, r[::-1]])
    prop_c = np.concatenate([c/4, -3*c[::-1]/4])
    prop_c += np.mean(c/4)
    print("Avg c/R:", np.mean(c))
    plt.plot(prop_r, prop_c, 'b', label="Flattened propeller")
    plt.plot(hub_x, hub_y, 'r', label="Propeller hub")
    # circle = plt.Circle((0, 0), 0.15, color='r')
    plt.xlim(0, 1)
    plt.ylim(-0.5, 0.5)
    plt.title("Propeller, R = " + str(R) + "m", fontsize="18")
    plt.ylabel("x/R", fontsize="16")
    plt.xlabel('y/R', fontsize="16")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    # ax.add_artist(circle)
    plt.show()

def plot_blade_cl():
    plt.figure()
    plt.plot([0, 0.1, 0.2, 0.5, 1.0], [0.84, 0.80, 0.64, 0.59, 0.55])
    plt.xlabel("r/R", fontsize="16")
    plt.ylabel("Blade cl", fontsize="16")
    plt.title("Prop Blade cl Distribution", fontsize="18")
    plt.show()

def design_opt_rpm(h=21000, plot=False, traj=False, opt=False):
    """
    Design a propeller by optimizing for efficiency on rpm for flight conditions
    given in template.mil and optional argument altitude
    """
    change_air_data(h)
    res = minimize_scalar(design_prop, bounds=(1100, 1600), method='bounded',
                          args=("temp_prop", traj, opt), options={'xatol': 2})
    if res.success:
        design_prop(res.x, "best_prop")
        if plot:
            print(-res.fun, res.x)
            plot_prop("best_prop")
        return res.x
    else:
        print("Unsuccessful optimization", res.x, res.fun)

if __name__ == "__main__":
    # change_prop_area(24)
    # design_opt_rpm(h=19800, plot=True, traj=True, opt=False)
    # change_air_data(20000)
    # design_prop(1100, "best_prop")
    plot_prop('best_prop')
    # plot_blade_cl()
    # print(-design_prop(1200, "test_prop"))

    # Sensitivity study to design RPM
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
