import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(1, 11, size=10)
y = np.random.randint(1, 11, size=10)

plt.scatter(x, y)
plt.title('Random Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()