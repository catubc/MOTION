import numpy as np
import matplotlib.pyplot as plt

data = np.float32([0,0.1,0.2,0.5,1.4,0.2,0.7,0.8,0.9,0.3])

data_array = []
for k in range(10):
    data_array.append(data[np.random.randint(0,10,10)])

plt.imshow(data_array, vmin=0, vmax=5.0, cmap='Reds')
plt.show()

