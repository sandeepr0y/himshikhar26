import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

y = np.maximum(0, x)

plt.plot(x, y)
plt.title("ReLU Function")
plt.grid(True)
plt.show()