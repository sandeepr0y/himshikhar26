import numpy as np
import matplotlib.pyplot as plt

# Generate 1D arrays for x and y coordinates.
# np.arange(-2, 3, 1) creates an array [-2, -1, 0, 1, 2].
x = np.arange(-2,3,1)
y = np.arange(-2,3,1)


# Create a 2D grid from the 1D x and y arrays.
# X will contain all x-coordinates, Y will contain all y-coordinates.
X,Y = np.meshgrid(x,y)



# Create a figure with 3 subplots for original, T1-transformed, and T2-transformed grids.
# fig, axs = plt.subplots(1,3,figsize=(18,6))

# --- Plot 1: Original Grid ---
# Plot horizontal lines (red)
plt.plot(X,Y,'r')

# Plot vertical lines (blue, dashed). Transposing X and Y makes the columns into rows.
plt.plot(X.T,Y.T,'b--')

# Plot corner points of the original square for reference.
# These points are (2,2), (2,-2), (-2,2), (-2,-2).
plt.plot(2,2,'g*',ms=10)
plt.plot(2,-2,'y^',ms=10)
plt.plot(-2,2,'co',ms=10)
plt.plot(-2,-2,'ms',ms=10)

# Add dashed black lines for the x and y axes for clarity.
plt.axvline(x=0,
               linestyle='--',
               color='black',
               alpha=0.7)
plt.axhline(y=0,
               linestyle='--',
               color='black',
               alpha=0.7)

# Set the title and axis limits for the plot.
plt.title("Original Grid")
plt.xlim(-5,5)
plt.ylim(-5,5)

# Enable grid and ensure equal aspect ratio for accurate visualization.
plt.grid(True, color='gray', linewidth=0.5, alpha=0.3)

plt.gca().set_aspect('equal', adjustable='box')

# Show the plot.
plt.show()