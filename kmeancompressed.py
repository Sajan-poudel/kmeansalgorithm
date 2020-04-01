import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
from PIL import Image

class Kmeans:
    '''IN THIS CLASS I HAVE IMPLEMENTED KMEANS ALGORITHM'''
    
    def __init__(self, num_cluster, max_iter=100, random_iter=100):
        self.num_cluster = num_cluster
        self.max_iter = max_iter
        self.random_iter = random_iter
        
        
    def randominit_centroids(self, X):
        np.random.RandomState(self.random_iter)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.num_cluster]]
        return centroids
    
    def closest_cluster(self, X, centroids):
        all_distance = np.zeros((X.shape[0], self.num_cluster))
        for k in range(self.num_cluster):
            all_distance[:,k] = np.square(np.linalg.norm(X - centroids[k, :], axis=1))
        return np.argmin(all_distance, axis=1)
    
    def centroids_mean(self, X, idx):
        centroids = np.zeros((self.num_cluster, X.shape[1]))
        for k in range(self.num_cluster):
            centroids[k, :] = np.mean(X[idx == k, :], axis=0)
        return centroids
    
    def compute_error(self, X,idx, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.num_cluster):
            distance[idx == k] = np.linalg.norm(X[idx == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))
    
    def calculate(self, X):
        self.centroids = self.randominit_centroids(X)
        for i in range(self.random_iter):
            previous_centoids = self.centroids
            self.idx = self.closest_cluster(X, previous_centoids)
            self.centroids = self.centroids_mean(X, self.idx)
            if np.all(previous_centoids == self.centroids):
                break
        self.error = self.compute_error(X,self.idx,self.centroids)
    
    def predict(self, X):
        return self.closest_cluster(X, self.centroids)
    
def main():
    img = Image.open('./pokhara.jpg')
    img = np.asarray(img)
    X = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    kmeans = Kmeans(num_cluster=16)
    kmeans.calculate(X)
    
    img_compressed = np.array([list(kmeans.centroids[id]) for id in kmeans.idx])
    img_compressed = img_compressed.astype("uint8")
    img_compressed = img_compressed.reshape(img.shape[0], img.shape[1], img.shape[2])
    fig, ax = plt.subplots(1, 2, figsize = (12, 8))
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[1].imshow(img_compressed)
    ax[1].set_title('Compressed Image with 16 colors')
    for ax in fig.axes:
        ax.axis('off')
    plt.tight_layout();
    compressed_img = Image.fromarray(img_compressed)
    compressed_img.save("pokharacompressed.jpg")
# if __name__ == "__main__":
main()
    