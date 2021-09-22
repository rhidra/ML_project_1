import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def pca(x,y):
    color = ['b', 'c', 'k', 'r', 'g', 'y', 'lightblue', 'm', 'orange', 'gray', 'lightcoral']
    pca = PCA(3)
    fig = plt.figure() #original pca
    com = pca.fit_transform(x.cpu().detach())
    ax = fig.gca(projection='3d')
    # ax.scatter(com[:, 0], com[:, 1], com[:, 2],s=10, c=y.cpu(),cmap=plt.cm.Spectral)
    for i in range(10):
        ax.scatter(com[y.cpu() == i, 0], com[y.cpu() == i, 1], com[y.cpu() == i, 2], s=10, c=color[i], label=i)
    plt.legend(loc='upper left')
    plt.show()

    kmeans = KMeans(n_clusters=10)
    fig = plt.figure() #pca kmeans
    com = pca.fit_transform(x.cpu().detach())
    com_k = kmeans.fit_predict(com)
    ax = fig.gca(projection='3d')
    for i in range(10):
        ax.scatter(com[com_k == i, 0], com[com_k == i, 1], com[com_k == i, 2], s=10, c=color[i], label=i)
    plt.legend(loc='upper left')
    plt.show()

def tsne(x,y):
    color = ['b', 'c', 'k', 'r', 'g', 'y', 'lightblue', 'm', 'orange', 'gray', 'lightcoral']
    fig = plt.figure() #TSNE
    com = TSNE(n_components=2).fit_transform(x.cpu().detach())
    ax = fig.gca()
    for i in range(int(max(y)+1)):
        ax.scatter(com[y == i, 0], com[y == i, 1], s=10, c=color[i], label=i)
    plt.legend()
    plt.show()