import numpy as np
import matplotlib.pyplot as plt
import cv2
import pathlib

def get_app_root():
    return pathlib.Path(__file__).parent.parent.parent.resolve()


def get_source_image():
    app_root = get_app_root()
    # img_path = app_root / 'himshikhar26' / 'Spider_Img.jpg'
    img_path = app_root / 'himshikhar26' / 'Bird.jpg'
    return cv2.imread(img_path)


def plot_image(ax, image, title='Image', cmap=None):
    if cmap:
        ax.imshow(image, cmap=cmap)
    else:
        ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')


def rank_k_approximation(N, k, sourceImgRed, sourceImgGreen, sourceImgBlue, Ur, Sr, Vr, Ug, Sg, Vg, Ub, Sb, Vb):
    k1_n = np.zeros_like(sourceImgRed, dtype=float)
    k2_n = np.zeros_like(sourceImgGreen, dtype=float)
    k3_n = np.zeros_like(sourceImgBlue, dtype=float)

    for i in range(N//k):
        k1_n += (Ur[:,i:i+1] @ np.array([[Sr[i]]]) @ Vr[i:i+1,:])
        k2_n += (Ug[:,i:i+1] @ np.array([[Sg[i]]]) @ Vg[i:i+1,:])
        k3_n += (Ub[:,i:i+1] @ np.array([[Sb[i]]]) @ Vb[i:i+1,:])

    img1 = np.uint8(np.clip(np.dstack((k1_n, k2_n, k3_n)), 0, 255))
    return img1


def main():
    SourceImg = get_source_image()
    SourceImg = np.array(SourceImg)

    sourceImgRed = SourceImg[:,:,0]
    sourceImgGreen = SourceImg[:,:,1]
    sourceImgBlue = SourceImg[:,:,2]

    fig, axs = plt.subplots(2, 4, figsize=(18,5))

    plot_image(axs[0,0], SourceImg, title='Source Image')
    plot_image(axs[0,1], sourceImgRed, title='Red Channel', cmap='Reds')
    plot_image(axs[0,2], sourceImgGreen, title='Green Channel', cmap='Greens')
    plot_image(axs[0,3], sourceImgBlue, title='Blue Channel', cmap='Blues')

    # SVD for Red Channel
    Ur, Sr, Vr = np.linalg.svd(sourceImgRed.astype(float))

    # SVD for Green Channel
    Ug, Sg, Vg = np.linalg.svd(sourceImgGreen.astype(float))

    # SVD for Blue Channel
    Ub, Sb, Vb = np.linalg.svd(sourceImgBlue.astype(float))

    # Rank Approximation
    N = SourceImg.shape[1]

    plot_image(axs[1,0], SourceImg, title='Source Image')
    for idx, k in enumerate((2, 20, 200)):
        plot_image(
            axs[1,idx+1],
            rank_k_approximation(N, k, sourceImgRed, sourceImgGreen, sourceImgBlue, Ur, Sr, Vr, Ug, Sg, Vg, Ub, Sb, Vb),
            title=f'For N/{k}'
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()