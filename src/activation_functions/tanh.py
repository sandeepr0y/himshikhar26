# tanh (Hyperbolic tangent)

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


plt.plot(x, y)
plt.title("tanh (Hyperbolic tangent) Activation Function")
plt.grid(True)
plt.show()