import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
leak = 0.01

y = np.where(x<0, leak*x, x)

plt.plot(x, y)
plt.title("Leaky ReLU Function")
plt.grid(True)
plt.show()