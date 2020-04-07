import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minize

def fit_eff(x, A1, A2, h1, h2, v1, v2, T1, T2, dB1, dB2, c):
    A, h, v, T, dB = x
    return A1*A**A2 + h1*h**h2 + v1*v**v2 + T1*T**T2 + dB1*(dB)**dB2 + c

data = np.load("opt_sweep.npz")
areas = data['areas']
hs = data['hs']
vs = data['vs']
ts = data['ts']
dbetas = data['dbetas']
ps = data['ps']
effs = data['effs']
indep = [hs, vs, ts]

plt.figure(figsize=(10,7))
plt.title("Efficiencies")
plt.subplot(231)
plt.ylabel(r"$\eta$")
plt.xlabel(r"Propulsive Areas [$m^2$]")
plt.plot(areas, np.mean(effs, axis=(1,2,3,4)))
plt.subplot(232)
plt.xlabel("Altitude [km]")
plt.plot(hs/1000, np.mean(effs, axis=(0,2,3,4)))
plt.subplot(233)
plt.xlabel("Airspeed [m/s]")
plt.plot(vs, np.mean(effs, axis=(0,1,3,4)))
plt.subplot(234)
plt.ylabel(r"$\eta$")
plt.xlabel("Thrust [N]")
plt.plot(ts, np.mean(effs, axis=(0,1,2,4)))
plt.subplot(235)
plt.xlabel(r"$d\beta$")
plt.plot(dbetas, np.mean(effs, axis=(0,1,2,3)))

plt.figure(figsize=(10,7))
plt.title("Shaft Powers")
plt.subplot(231)
plt.ylabel("Shaft Power")
plt.xlabel(r"Propulsive Areas [$m^2$]")
plt.plot(areas, np.mean(ps, axis=(1,2,3,4)))
plt.subplot(232)
plt.xlabel("Altitude [km]")
plt.plot(hs/1000, np.mean(ps, axis=(0,2,3,4)))
plt.subplot(233)
plt.xlabel("Airspeed [m/s]")
plt.plot(vs, np.mean(ps, axis=(0,1,3,4)))
plt.subplot(234)
plt.ylabel(r"Shaft Power")
plt.xlabel("Thrust [N]")
plt.plot(ts, np.mean(ps, axis=(0,1,2,4)))
plt.subplot(235)
plt.xlabel(r"$d\beta$")
plt.plot(dbetas, np.mean(ps, axis=(0,1,2,3)))

plt.show()

opt = minimize()

# popt, pcov = curve_fit(fit_func, [hs, vs, ts], )
