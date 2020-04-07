import matplotlib.pyplot as plt
import numpy as np
from numpy import radians as r

CL0 = 0.4
CL_a = 6.2832
CLmin = -0.4
CLmax = 1.3
CD0 = 0.02
CD2u = 0.008
CD2l = 0.006
CLCD0 = 0.30
REref = 100000
REexp = -0.5
alphas = np.linspace(r(-10), r(10), 200)
v = 20
a = 298.4
Re = 150000
beta = (1-(v/a)**2)**(1/2)
CL = np.zeros(200)
CD = np.zeros(200)
for i, alp in enumerate(alphas):
    CL[i] = (alp*CL_a + CL0)/beta
    stall = False
    if CL[i] > CLmax:
        stall = True
        CL[i] = CLmax
    elif CL[i] < CLmin:
        stall = True
        CL[i] = CLmin
    CD2 = CD2u if CL[i] > CLCD0 else CD2l
    CD[i] = (CD0 + CD2*(CL[i]-CLCD0)**2) * (Re/REref)**REexp
    if stall:
        CD[i] += 2*(np.sin(alp - (CLCD0-CL0)/CL_a))**2

plt.plot(alphas, CL)
plt.show()
plt.plot(alphas, CD)
plt.show()
