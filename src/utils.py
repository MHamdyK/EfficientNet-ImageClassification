# src/utils.py

import matplotlib.pyplot as plt

def show_image(image, title=""):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()
