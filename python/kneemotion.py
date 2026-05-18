import numpy as np
import matplotlib.pyplot as plt
from wav import to_wav


t = np.linspace(0, 2, 1000)

def sin_rectified(t):
    return np.sin(t)*0.5+0.5

f = 1
ft = t*2*np.pi*f
v = 2*sin_rectified(ft) + sin_rectified(ft*2)

plt.plot(t, v)
plt.show()





