import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

y = 1 / (1 + np.exp(-x))


plt.plot(x, y)
plt.title("Sigmoid Function")
plt.grid(True)
plt.show()