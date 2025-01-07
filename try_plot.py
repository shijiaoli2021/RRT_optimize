import matplotlib.pyplot as plt
from data.trajectoryData import *
import numpy as np
import exp1

for i in range(1, 10):
    plt.subplot(3, 3, i)
    exp1.plotMap(plt, area)
    tra = np.array(trajectory[i+3])
    plt.plot(tra[:, 0], tra[:, 1])

plt.show()