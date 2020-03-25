
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
x, y_true = make_blobs(n_samples=500, centers=5,
                       cluster_std=0.60)
plt.figure()
plt.scatter(x[:,0],x[:,1],c=y_true,s=25)
from sklearn.cluster import KMeans
#we can also put init='k-means++' in p to get more accurate clusters
p=KMeans(n_clusters=4)
p.fit(x)
y_pred=p.predict(x)
centers=p.cluster_centers_
plt.figure()
plt.scatter(x[:,0],x[:,1],c=y_pred,s=25)
plt.scatter(centers[:,0],centers[:,1],c='r',s=25)
center=p.cluster_centers_
plt.scatter(center[:,0],center[:,1],c='black',s=200,alpha=0.5)


