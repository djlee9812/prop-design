import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import aerosandbox.library.atmosphere as atmo
from scipy.interpolate import interp1d
from lmfit import Model, Parameter

def get_air_data(alt):
    """
    Read and extrapolate air data in air_data.csv and return air density,
    dynamic viscosity, and speed of sound at the given altitude.
    """
    df = pd.read_csv("air_data.csv")
    h = df['h']
    a = df['a']
    rho = df['rho']
    mu = df['mu']*1e5
    # linear for a, mu; quadratic for rho, nu
    f_rho = interp1d(h, rho, kind="quadratic")
    f_mu = interp1d(h, mu)
    f_a = interp1d(h, a)
    return np.round(f_rho(alt),4), np.round(f_mu(alt),4), np.round(f_a(alt),1)

def get_atmo_data(alt):
    """
    Peter's fitted atmosphere model good for <40km altitude
    ~1000x faster than interpolating
    """
    P = atmo.get_pressure_at_altitude(alt)
    rho = atmo.get_density_at_altitude(alt)
    T = atmo.get_temperature_at_altitude(alt)
    mu = atmo.get_viscosity_from_temperature(T)*1e5
    a = atmo.get_speed_of_sound_from_temperature(T)
    return rho, mu, a

# def fitted_air_data(alt):
#     if alt < 11000:
#         T = 273.15 + 15.04 - 0.00649*alt
#         p = 101.29 * (T/288.08) ** 5.256
#     elif alt < 25000:
#         T = 273.15 - 56.46
#         p = 22.65 * np.exp(1.73 - 0.000157*alt)
#     else:
#         T = 273.15 -131.21 + 0.00299*alt
#         p = 2.488 * (T/216.6) ** -11.388
#     rho = p / (.2869 * T)
#     a = np.sqrt(1.4 * 287 * T)

def fit_T(alt, b1, c1, b2, b3, c3, b4, c4):
    T = np.zeros(alt.shape)
    for i, h in enumerate(alt):
        if h <= 11000:
            T[i] = b1 + c1*h
        elif h <= 20000:
            T[i]= b2
        elif h <= 32000:
            T[i] = b3 + c3*h
        else:
            T[i] = b4 + c4*h
    return T

def fit_p(alt, b1, c1, exp1, b2, c2, exp2, b3, c3, exp3, b4, c4, exp4):
    p = np.zeros(alt.shape)
    for i, h in enumerate(alt):
        if h <= 11000:
            T = 288.15 - 0.0065*h
            p[i] = b1 * (T/c1) ** exp1
        elif h <= 20000:
            T = 216.65
            p[i] = b2 * np.exp(c2 + exp2*h)
        elif h <= 32000:
            T = 196.65 + 0.001*h
            p[i] = b3 * (T/c3) ** exp3
        else:
            T = 148.236 + 0.002554*h
            p[i] = b4 * (T/c4) ** exp4
    return p

def fit_rho(alt, crho):
    rho = np.zeros(alt.shape)
    for i, h in enumerate(alt):
        if h < 11000:
            T = 273.15 + 15.04 - 0.00649*h
            p = 101.29 * (T/288.08) ** 5.256
        elif h < 20000:
            T = 273.15 - 56.46
            p = 22.65 * np.exp(1.73 - 0.000157*h)
        elif h < 32000:
            T = 273.15 -131.21 + 0.00299*h
            p = 2.488 * (T/216.6) ** -11.388
        else:
            pass
        rho[i] = p / (crho * T)
        a = np.sqrt(1.4 * 287 * T)
    return rho

def change_air_data(alt):
    """
    Rewrite qcon.def which specifies air_data for qmil and qprop to the given
    altitude
    """
    rho, mu, a = get_atmo_data(alt)
    # print(rho, mu, a)
    data = []
    data.append(str(rho) + " ! rho  kg/m^3; alt: " + str(alt) + "m\n" )
    data.append(str(mu) + "E-5 ! mu   kg/m-s\n")
    data.append(str(a) + " ! a    m/s")
    with open('qcon.def', 'w') as file:
        file.writelines(data)
    return rho, mu, a

def plot_air_data():
    """
    Plot air density, dynamic viscosity, and speed of sound as a function of
    altitude, extrapolating data from air_data.csv
    """
    alts = np.arange(0, 50000, 1000)
    rhos = np.zeros(len(alts))
    mus = np.zeros(len(alts))
    ass = np.zeros(len(alts))
    rhos2 = np.zeros(len(alts))
    mus2 = np.zeros(len(alts))
    ass2 = np.zeros(len(alts))
    for n, h in enumerate(alts):
        rhos[n], mus[n], ass[n] = get_air_data(h)
        rhos2[n], mus2[n], ass2[n] = get_atmo_data(h)

    fig = plt.figure(figsize=(10,5))
    plt.title("Standard Atmosphere")
    plt.subplot(131)
    plt.plot(rhos, alts/1000)
    plt.plot(rhos2, alts/1000)
    plt.axhline(color="0.5", linestyle=":")
    plt.axvline(color="0.5", linestyle=":")
    plt.ylabel("Altitude (km)")
    plt.xlabel(r"$\rho$ (kg/$m^3$)")
    plt.subplot(132)
    plt.plot(mus, alts/1000)
    plt.plot(mus2, alts/1000)
    plt.xlabel(r"$\mu$")
    plt.axhline(color="0.5", linestyle=":")
    plt.subplot(133)
    plt.plot(ass, alts/1000)
    plt.plot(ass2, alts/1000)
    plt.xlabel("a (m/s)")
    plt.axhline(color="0.5", linestyle=":")
    plt.show()

if __name__ == "__main__":
    alt = 22000 if len(sys.argv) <= 1 else int(sys.argv[1])
    change_air_data(alt)
    # plot_air_data()
    # df = pd.read_csv("air_data.csv")
    # h = df['h'].to_numpy()
    # a = df['a'].to_numpy()
    # rho = df['rho'].to_numpy()
    # mu = df['mu'].to_numpy()
    # p = df['p'].to_numpy()
    # T = df['T'].to_numpy()

    # start = time.time()
    # for i in range(10000):
    #     get_atmo_data(20000+2*i)
    # print(time.time()-start)

    # T_model = Model(fit_T)
    # T_params = T_model.make_params(b1=288.15, c1=-0.0065, b2=216.65, b3=196.65, c3=0.001, b4=149.4, c4=0.0025)
    # T_result = T_model.fit(T, params, alt=h)
    # print(T_result.fit_report())
    # Rsquared = 1 - result.residual.var()/np.var(T)
    # print(Rsquared)
    # p_model = Model(fit_p)
    # p_params = p_model.make_params(b1=101300, c1=288, exp1=5.26, b2=19900, c2=1.9, exp2=-0.00016,
    #                                b3=2000, c3=220, exp3=-11, b4=1500, c4=180, exp4=-13)
    # p_result = p_model.fit(p, p_params, alt=h)
    # print(p_result.fit_report())
    # print(1 - p_result.residual.var()/np.var(p))
    # rho_model = Model(fit_rho)
    # params = rho_model.make_params(crho=1)
    # result = T_model.fit(T, params, alt=h)
    # print(result.fit_report())
    # Rsquared = 1 - result.residual.var()/np.var(T)
    # print(Rsquared)
