import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.interpolate import interp1d

def get_air_data(alt):
    """
    Read and extrapolate air data in air_data.csv and return air density,
    dynamic viscosity, and speed of sound at the given altitude.
    """
    df = pd.read_csv("air_data.csv")
    h = df['h']
    a = df['a']
    rho = df['rho/rho0'] * 1.225
    nu = df['nu']
    mu = nu * rho
    # linear for a, mu; quadratic for rho, nu
    f_rho = interp1d(h, rho, kind="quadratic")
    f_mu = interp1d(h, mu)
    f_a = interp1d(h, a)
    return np.round(f_rho(alt),4), np.round(f_mu(alt),4), np.round(f_a(alt),1)

    # i = h[h == alt].index[0]
    # return rho[i], mu[i], a[i]

def change_air_data(alt):
    """
    Rewrite qcon.def which specifies air_data for qmil and qprop to the given
    altitude
    """
    rho, mu, a = get_air_data(alt)
    # print(rho, mu, a)
    data = []
    data.append(str(rho) + " ! rho  kg/m^3; alt: " + str(alt) + "m\n" )
    data.append(str(mu) + "E-5 ! mu   kg/m-s\n")
    data.append(str(a) + " ! a    m/s")
    with open('qcon.def', 'w') as file:
        file.writelines(data)

def plot_air_data():
    """
    Plot air density, dynamic viscosity, and speed of sound as a function of
    altitude, extrapolating data from air_data.csv
    """
    alts = np.arange(-2000, 50000, 1000)
    rhos = np.zeros(len(alts))
    mus = np.zeros(len(alts))
    ass = np.zeros(len(alts))
    for n, h in enumerate(alts):
        rhos[n], mus[n], ass[n] = get_air_data(h)

    fig = plt.figure(figsize=(10,5))
    plt.title("Standard Atmosphere")
    plt.subplot(131)
    plt.plot(rhos, alts/1000)
    plt.axhline(color="0.5", linestyle=":")
    plt.axvline(color="0.5", linestyle=":")
    plt.ylabel("Altitude (km)")
    plt.xlabel(r"$\rho$ (kg/$m^3$)")
    plt.subplot(132)
    plt.plot(mus, alts/1000)
    plt.xlabel(r"$\mu$")
    plt.axhline(color="0.5", linestyle=":")
    plt.subplot(133)
    plt.plot(ass, alts/1000)
    plt.xlabel("a (m/s)")
    plt.axhline(color="0.5", linestyle=":")
    plt.show()

if __name__ == "__main__":
    alt = 22000 if len(sys.argv) <= 1 else sys.argv[1]
    change_air_data(alt)
    # plot_air_data()
