import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


path = "/app/himshikhar26/Lab1_Img.jpg"

img1 = cv2.imread(path)


img = cv2.cvtColor(img1,
                   cv2.COLOR_BGR2RGB)

h, w = img.shape[:2]

x = np.arange(0,w)
y = np.arange(0,h)

X, Y = np.meshgrid(x,y)

a = X.flatten()
b = Y.flatten()

points = np.vstack((a,b))

cx = w/2
cy = h/2

print(f"Center = ({cx},{cy})\n")

theta = np.radians(180)

T = np.array([[np.cos(theta), -np.sin(theta)],
               [np.sin(theta),  np.cos(theta)]])


# ==========================================================
# Part F : Create Empty Output Images
# ==========================================================

rot_theta_img  = np.zeros_like(img)


for y_new in range(h):

    for x_new in range(w):

        # --------------------------------------------------
        # Shift Coordinates to Center
        # --------------------------------------------------

        x_shift = x_new - cx
        y_shift = y_new - cy

        # ==================================================
        # Rotation by theta Degrees
        # ==================================================

        source_theta = T @ np.array([[x_shift],
                                      [y_shift]])
        # print(f"Source Theta = \n{source_theta}\n")
        # print(source_theta[0,0], source_theta[1,0])
        # print(source_theta[0][0], source_theta[1][0])
        # exit()
        # rot_theta_img[y_new, x_new] = img[round(source_theta[0][0]), round(source_theta[1][0])]

        xs = int(round(source_theta[0,0] + cx))
        ys = int(round(source_theta[1,0] + cy))

        if (0 <= xs < w) and (0 <= ys < h):
            rot_theta_img[y_new, x_new] = img[ys, xs]

# ==========================================================
# Visualization
# ==========================================================

fig, axs = plt.subplots(1, 2)

# ==========================================================
# Plot 1 : Original Image
# ==========================================================

axs[0].imshow(img)
axs[0].set_title("Original Image")
axs[0].axis('off')

# ==========================================================
# Plot 2 : T1 = 90 Degrees
# ==========================================================

axs[1].imshow(rot_theta_img)
axs[1].set_title(f"T1 = {int(np.degrees(theta))}°")
axs[1].axis('off')

plt.tight_layout()
plt.show()