import numpy as np
import matplotlib.pyplot as plt
from pysvm import svm
from wav import to_wav
from sin_rectified import sin_r

def inverse_park(q, d, theta):
    alpha = d*np.cos(theta) - q*np.sin(theta)
    beta = d*np.sin(theta) + q*np.cos(theta)
    return alpha, beta  

def inverse_clarke(alpha, beta):
    Va = alpha
    Vc = -alpha/2 - np.sqrt(3)/2*beta
    Vb = -alpha/2 + np.sqrt(3)/2*beta
    return -Va, -Vb, -Vc



t = np.linspace(0, 50e-3, 4000)
ft = t*2*np.pi*20
Vm = (2*sin_r(ft) - sin_r(ft*2))*0.5
Fhz = 1000
theta = np.mod(t*Fhz*2*np.pi + np.pi, 2*np.pi) - np.pi

alpha,beta = inverse_park(Vm, 0, theta)
Va, Vb, Vc = inverse_clarke(alpha, beta)
tA, tB, tC, sector_svmf = svm(alpha, beta)
Va_svm = tA
Vb_svm = tB
Vc_svm = tC

"""
This definition of sector is consistent with the SVM sector definition.
"""
sector_v = t*0
for i in range(len(t)):
    sector = 0
    if(Va[i] > Vc[i] and Vc[i] > Vb[i]):
        sector = 6
    elif(Vc[i] > Va[i] and Va[i] > Vb[i]):
        sector = 5
    elif(Vc[i] > Vb[i] and Vb[i] > Va[i]):
        sector = 4
    elif(Vb[i] > Vc[i] and Vc[i] > Va[i]):
        sector = 3
    elif(Vb[i] > Va[i] and Va[i] > Vc[i]):
        sector = 2
    elif(Va[i] > Vb[i] and Vb[i] > Vc[i]):
        sector = 1
    sector_v[i] = sector

Gah = t*0
Gal = t*0
Gbh = t*0
Gbl = t*0
Gch = t*0
Gcl = t*0

for i in range(int(len(t)/2),len(t)):
    if(sector_v[i] == 1):
        Gah[i] = 1
        Gal[i] = 0
        Gbh[i] = 0
        Gbl[i] = 0
        Gch[i] = 0
        Gcl[i] = 1
    elif(sector_v[i] == 2):
        Gah[i] = 0
        Gal[i] = 0
        Gbh[i] = 1
        Gbl[i] = 0
        Gch[i] = 0
        Gcl[i] = 1
    elif(sector_v[i] == 3):
        Gah[i] = 0
        Gal[i] = 1
        Gbh[i] = 1
        Gbl[i] = 0
        Gch[i] = 0
        Gcl[i] = 0
    elif(sector_v[i] == 4):
        Gah[i] = 0
        Gal[i] = 1
        Gbh[i] = 0
        Gbl[i] = 0
        Gch[i] = 1
        Gcl[i] = 0
    elif(sector_v[i] == 5):
        Gah[i] = 0
        Gal[i] = 0
        Gbh[i] = 0
        Gbl[i] = 1
        Gch[i] = 1
        Gcl[i] = 0
    elif(sector_v[i] == 6):
        Gah[i] = 1
        Gal[i] = 0
        Gbh[i] = 0
        Gbl[i] = 1
        Gch[i] = 0
        Gcl[i] = 0

prev_sector = 0
for i in range(len(t)):
    if(sector_v[i] != prev_sector):
        prev_sector = sector_v[i]
        if(i-1 > 0 and i+1 < len(t)-1):
            Gah[i-1:i+2] = 0
            Gal[i-1:i+2] = 0
            Gbh[i-1:i+2] = 0
            Gbl[i-1:i+2] = 0
            Gch[i-1:i+2] = 0
            Gcl[i-1:i+2] = 0


to_wav(t, Va, 'Va.wav', 1)
to_wav(t, Vb, 'Vb.wav', 1)
to_wav(t, Vc, 'Vc.wav', 1)
to_wav(t, Gah, 'Gah.wav', 1)
to_wav(t, Gal, 'Gal.wav', 1)
to_wav(t, Gbh, 'Gbh.wav', 1)
to_wav(t, Gbl, 'Gbl.wav', 1)
to_wav(t, Gch, 'Gch.wav', 1)
to_wav(t, Gcl, 'Gcl.wav', 1)
to_wav(t, sector_v, 'sector_v.wav', 1)


fig, ax = plt.subplots(3, 1)
ax[0].plot(t, Va)
ax[0].plot(t, Vb)
ax[0].plot(t, Vc)
ax[1].plot(t, Va_svm)
ax[1].plot(t, Vb_svm)
ax[1].plot(t, Vc_svm)

ax[2].plot(t, Gah)
ax[2].plot(t, Gal)
ax[2].plot(t, Gbh+3)
ax[2].plot(t, Gbl+3)
ax[2].plot(t, Gch+6)
ax[2].plot(t, Gcl+6)

# ax.plot(t, theta)
# ax.plot(t, sector_v)
# ax.plot(t, sector_svmf)

# Define colors for each sector using RGB tuples (values between 0 and 1)
sector_colors = {
    1: (1.0, 0.0, 0.0),    # red
    2: (0.0, 1.0, 0.0),    # green
    3: (0.0, 0.0, 1.0),    # blue
    4: (0.5, 0.0, 0.5),    # purple
    5: (1.0, 0.5, 0.0),    # orange
    6: (0.0, 1.0, 1.0)     # cyan
}

start_region = 0
previous_sector = 0
for i in range(len(sector_v)):
    if sector_v[i] != previous_sector or i == len(sector_v) - 1:
        end_region = i
        if(previous_sector >= 1 and previous_sector <= 6):
            ax[0].axvspan(t[start_region], t[end_region], alpha=0.2, color=sector_colors[previous_sector])
            ax[1].axvspan(t[start_region], t[end_region], alpha=0.2, color=sector_colors[previous_sector])
        previous_sector = sector_v[i]
        start_region = i

ax[0].legend(['Va', 'Vb', 'Vc'])
ax[1].legend(['Va', 'Vb', 'Vc'])
ax[2].legend(['Gah', 'Gal', 'Gbh', 'Gbl', 'Gch', 'Gcl'])
plt.show()


