
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MeanShift
centers=([1,1],[5,7],[10,12])
X,_=make_blobs(n_samples=10000,cluster_std=1,random_state=42,centers=centers) 
plt.scatter(X[:,0],X[:,1])
plt.show()
r=MeanShift()
r.fit(X)
labels=r.predict(X)
#or labels=r.labels_
cluster_centers=r.cluster_centers_
n=len(np.unique(labels))
print("number of clusters:",n)
colors=['r.','b.','g.','k.','m.','c.','y.']
#print(colors)
#print(labels)
for i in range(len(X)):
    plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=10)
    
plt.scatter(cluster_centers[:,0],cluster_centers[:,1],marker='o',s=100,zorder=10,c='y'*3)
plt.show()    
