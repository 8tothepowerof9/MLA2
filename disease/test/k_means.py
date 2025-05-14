import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def kmeans_lab_segmentation(image_path, k=3):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
    h, w, _ = image_lab.shape

    # Flatten LAB image for clustering
    lab_flat = image_lab.reshape((-1, 3))

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(lab_flat)
    labels = kmeans.labels_
    clustered = labels.reshape((h, w))

    # Visualize each cluster separately
    cluster_visuals = []
    for i in range(k):
        mask = (clustered == i)
        cluster_img = np.zeros_like(image_rgb)
        cluster_img[mask] = image_rgb[mask]
        cluster_visuals.append(cluster_img)

    # Plot results
    plt.figure(figsize=(16, 4))
    plt.subplot(1, k + 1, 1)
    plt.imshow(image_rgb)
    plt.title("Original")
    plt.axis("off")

    for i, cluster_img in enumerate(cluster_visuals):
        plt.subplot(1, k + 1, i + 2)
        plt.imshow(cluster_img)
        plt.title(f"Cluster {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Example usage
kmeans_lab_segmentation("data/train_images/bacterial_leaf_blight/100330.jpg", k=4)
