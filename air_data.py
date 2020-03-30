import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import sys

def get_air_data(alt):
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
    rho, mu, a = get_air_data(alt)
    print(rho, mu, a)
    data = []
    data.append(str(rho) + " ! rho  kg/m^3; alt: " + str(alt) + "m\n" )
    data.append(str(mu) + "E-5 ! mu   kg/m-s\n")
    data.append(str(a) + " ! a    m/s")
    with open('qcon.def', 'w') as file:
        file.writelines(data)

if __name__ == "__main__":
    alt = 25000 if len(sys.argv) <= 1 else sys.argv[1]
    change_air_data(alt)
