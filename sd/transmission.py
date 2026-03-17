import numpy as np
import cv2
from matplotlib import pyplot as plt

def estimate_A(img):
    # img: float32, range [0,1]
    dark = np.min(img, axis=2)

    flat_dark = dark.flatten()
    num_pixels = int(0.001 * len(flat_dark))
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]

    brightest = img.reshape(-1, 3)[indices]
    A = np.max(brightest, axis=0)

    return A

def estimate_transmission(img, A, omega=0.95):
    normed = img / A
    dark = np.min(normed, axis=2)
    t = 1 - omega * dark
    return t

def refine_transmission(t):
    t_refined = cv2.GaussianBlur(t, (15,15), 0)
    return t_refined

def recover_image(img, t, A, t0=0.1):
    t = np.maximum(t, t0)  # avoid division explosion
    J = (img - A) / t[..., None] + A
    J = np.clip(J, 0, 1)
    return J

def underwater_restoration(image_path):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    A = estimate_A(img)
    t = estimate_transmission(img, A)
    print(t.shape)
    t = refine_transmission(t)
    print(t.shape)
    J = recover_image(img, t, A)

    return (J * 255).astype(np.uint8)

if __name__ == '__main__':
    path = '/media/ty/ADATA SE880/data/UIE-dataset/UIEBD/test/image/717.jpg'
    r = underwater_restoration(path)
    img = cv2.imread(path)
    plt.subplot(1, 2, 1)
    plt.imshow(img[:, :, ::-1])
    plt.subplot(1, 2, 2)
    plt.imshow(r)
    plt.show()
