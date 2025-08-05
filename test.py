import matplotlib.pyplot as plt
from umap.parametric_umap import ParametricUMAP
from tensorflow.keras.datasets import mnist

# データセットの準備
(train_images, Y_train), (test_images, Y_test) = mnist.load_data()
train_images = train_images.reshape((train_images.shape[0], -1))/255.
test_images = test_images.reshape((test_images.shape[0], -1))/255.

# 埋め込み実行
embedder = ParametricUMAP(verbose=True)
embedding = embedder.fit_transform(train_images)

# Plot the embedding
fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=Y_train.astype(int),
    cmap="tab10",
    s=0.1,
    alpha=0.5,
    rasterized=True,
)
ax.axis('equal')
ax.set_title("Parametric UMAP embedding", fontsize=20)
plt.colorbar(sc, ax=ax)

# Save the plot to a file
fig.savefig('parametric_umap_embedding.png')

