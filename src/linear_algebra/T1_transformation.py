import numpy as np
import matplotlib.pyplot as plt

T1 = np.array([[2,0],
               [0,1]])

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# Generate 1D arrays for x and y coordinates.
# np.arange(-2, 3, 1) creates an array [-2, -1, 0, 1, 2].
x = np.arange(-2,3,1)
y = np.arange(-2,3,1)


# Create a 2D grid from the 1D x and y arrays.
# X will contain all x-coordinates, Y will contain all y-coordinates.
X,Y = np.meshgrid(x,y)

# --- Plot 1: Original Grid ---
# Plot horizontal lines (red)
axs[0].plot(X,Y,'r')

# Plot vertical lines (blue, dashed). Transposing X and Y makes the columns into rows.
axs[0].plot(X.T,Y.T,'b--')

# Plot corner points of the original square for reference.
# These points are (2,2), (2,-2), (-2,2), (-2,-2).
axs[0].plot(2,2,'g*',ms=10)
axs[0].plot(2,-2,'y^',ms=10)
axs[0].plot(-2,2,'co',ms=10)
axs[0].plot(-2,-2,'ms',ms=10)

# Add dashed black lines for the x and y axes for clarity.
axs[0].axvline(x=0,
               linestyle='--',
               color='black',
               alpha=0.7)
axs[0].axhline(y=0,
               linestyle='--',
               color='black',
               alpha=0.7)

# Set the title and axis limits for the plot.
axs[0].set_title("Original Grid")
axs[0].set_xlim(-5,5)
axs[0].set_ylim(-5,5)

# Enable grid and ensure equal aspect ratio for accurate visualization.
axs[0].grid(True, color='gray', linewidth=0.5, alpha=0.3)
axs[0].set_aspect('equal')

# Flatten the 2D X and Y grid arrays into 1D arrays.
# 'a' will contain all x-coordinates, 'b' will contain all y-coordinates.
a = X.flatten()
b = Y.flatten()

# Stack 'a' and 'b' vertically to create a 2xN matrix of points.
# Each column in 'points' represents a coordinate (x, y) from the grid.
points = np.vstack((a,b))

# Apply the transformation T1 to all grid points.
# The '@' operator performs matrix multiplication.
transformed_t1 = T1 @ points

# Reshape the transformed x-coordinates back to the original grid shape.
X_t1 = transformed_t1[0,:].reshape(X.shape)
# print(f"X_T1 = \n{X_t1}\n")

# Reshape the transformed y-coordinates back to the original grid shape.
Y_t1 = transformed_t1[1,:].reshape(Y.shape)
# print(f"Y_T1 = \n{Y_t1}\n")


# --- Plot 2: Grid after T1 ---
# Plot horizontal lines of the T1-transformed grid (red).
axs[1].plot(X_t1,Y_t1,'r')

# Plot vertical lines of the T1-transformed grid (blue, dashed).
axs[1].plot(X_t1.T,Y_t1.T,'b--')

# Transform the original corner points using T1 and plot them.
# Original corners were (2,2), (2,-2), (-2,2), (-2,-2).
t1_corner_1 = T1 @ np.array([[2],[2]])
t1_corner_2 = T1 @ np.array([[2],[-2]])
t1_corner_3 = T1 @ np.array([[-2],[2]])
t1_corner_4 = T1 @ np.array([[-2],[-2]])



axs[1].plot(t1_corner_1[0],t1_corner_1[1],'g*',ms=10)
axs[1].plot(t1_corner_2[0],t1_corner_2[1],'y^',ms=10)
axs[1].plot(t1_corner_3[0],t1_corner_3[1],'co',ms=10)
axs[1].plot(t1_corner_4[0],t1_corner_4[1],'ms',ms=10)

# Add dashed black lines for the x and y axes.
axs[1].axvline(x=0,
               linestyle='--',
               color='black',
               alpha=0.7)
axs[1].axhline(y=0,
               linestyle='--',
               color='black',
               alpha=0.7)

# Set the title and axis limits for the plot.
axs[1].set_title("Grid after T1")
axs[1].set_xlim(-5,5)
axs[1].set_ylim(-5,5)

# Enable grid and ensure equal aspect ratio.
axs[1].grid(True, color='gray', linewidth=0.5, alpha=0.3)
axs[1].set_aspect('equal')

plt.show()