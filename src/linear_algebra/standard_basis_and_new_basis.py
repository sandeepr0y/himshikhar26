import numpy as np
import matplotlib.pyplot as plt

v = np.array([[3],[2]])
s = np.array([[1,1],[-1,1]])
s_inv = np.linalg.inv(s)
v_new = s_inv @ v

# Create a figure with two subplots side-by-side for visualization

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Standard Basis Representation

v = np.array([[3],[2]])
print(v[1][0])

# Plot vector v in the standard basis (red arrow)
axs[0].quiver(0, 0, v[0], v[1],
              angles='xy',
              scale_units='xy',
              scale=1,
              color='red',
              label='v')

# Plot standard basis vector e1 (black arrow)
axs[0].quiver(0, 0, 1, 0,
              angles='xy',
              scale_units='xy',
              scale=1,
              color='black',
              label='e1')

# Plot standard basis vector e2 (blue arrow)
axs[0].quiver(0, 0, 0, 1,
              angles='xy',
              scale_units='xy',
              scale=1,
              color='blue',
              label='e2')

# Add dashed lines for x and y axes
axs[0].axvline(x=0,
               linestyle='--',
               color='black',
               alpha=0.7)

axs[0].axhline(y=0,
               linestyle='--',
               color='black',
               alpha=0.7)

axs[0].set_title("Vector in Standard Basis")

# Set x and y axis limits for better visualization
axs[0].set_xlim(-4, 4)
axs[0].set_ylim(-4, 4)

axs[0].grid(True, color='gray', linewidth=0.5, alpha=0.3) # Add a grid to the plot
axs[0].legend() # Display the legend for vectors
axs[0].set_aspect('equal') # Ensure equal scaling for axes



# Subplot 2: New Basis Representation

# Plot the coordinate vector [v]_B in the new basis (red arrow)
axs[1].quiver(0, 0, v_new[0], v_new[1],
              angles='xy',
              scale_units='xy',
              scale=1,
              color='red',
              label='[v]$_B$')

# Plot the first basis vector b1 from matrix S (green arrow)
axs[1].quiver(0, 0, s[0][0], s[1][0],
              angles='xy',
              scale_units='xy',
              scale=1,
              color='green',
              label='b1')

# Plot the second basis vector b2 from matrix S (magenta arrow)
axs[1].quiver(0, 0, s[0][1], s[1][1],
              angles='xy',
              scale_units='xy',
              scale=1,
              color='magenta',
              label='b2')

# Add dashed lines for x and y axes
axs[1].axvline(x=0,
               linestyle='--',
               color='black',
               alpha=0.7)

axs[1].axhline(y=0,
               linestyle='--',
               color='black',
               alpha=0.7)

axs[1].set_title("Coordinates in Basis B")

# Set x and y axis limits for better visualization
axs[1].set_xlim(-4, 4)
axs[1].set_ylim(-4, 4)

axs[1].grid(True, color='gray', linewidth=0.5, alpha=0.3) # Add a grid to the plot
axs[1].legend() # Display the legend for vectors
axs[1].set_aspect('equal') # Ensure equal scaling for axes

# Adjust layout to prevent overlapping titles and labels
plt.tight_layout()
# Display both plots
plt.show()
