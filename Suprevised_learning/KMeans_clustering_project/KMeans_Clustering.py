import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Load the dataset
url =r"C:\Users\nisar\Desktop\AI Ml\Mall_Customers.csv"
data = pd.read_csv(url)
#print(data.head())

#select the relevant features for clustering

#We'll use Annual Income and Spending Score
#Now you can select columns easily

x=data[["Annual Income (k$)","Spending Score (1-100)"]]
#print(x.head())

# step 4 Feature Scaling (import for Kmeans)

scaler = StandardScaler()
x_scaled= scaler.fit_transform(x)
print(x)
print(x_scaled)#all the data are betn 0 and 1

#step5:use Elbow Method to choose optional K
#Elbow Method

inertia=[]#[]list

K_range=range(1,11)# how the range function work 1 to n we get the n-1

for k in K_range:# this work for 10 time
    kmeans=KMeans(n_clusters =k, random_state=42)
    kmeans.fit(x_scaled)
    inertia.append(kmeans.inertia_)

#print values for refrences
print("K vs Inertia:\n")
for k,val in zip(K_range,inertia):
    print(f"k={k}-> Inertia = {val:.2f}")


#plot the Elbow graph
plt.figure(figsize=(8,5))
plt.plot(K_range,inertia,marker='o',color='purple')
plt.title("Elbow Method for Optional K")
plt.xlabel("no of clusters(k)")
plt.ylabel("Inertia")
plt.grid(True)

#Annolate each point with (k,inertia)
for k, val in zip(K_range,inertia):
    plt.text(k, val, f"k={k}\n{val:.0f}",ha='center', va= 'bottom', fontsize=8)

plt.show()



# step 6: apply the optimal K (say 5) 
#how we say that the k=5 is perfect the difference betn 5 and 6 is min
# as compare to all the others
kmeans = KMeans(n_clusters=5
                , random_state =42)
clusters = kmeans.fit_predict(x_scaled)

# dat the cluster label to the original data
data["cluster"]= clusters
print(data)


# step 7:visulaize the cluster
plt.figure(figsize=(8,6))
plt.scatter(x_scaled[:,0],x_scaled[:,1],c=clusters,cmap='viridis',s=50)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],
            s=200,c='red',marker='x', label= 'Centroids')
plt.title('KMeans Clustering Results')
plt.xlabel("annual Income (scaled)")
plt.ylabel("Spending Score(scaled)")
plt.legend()
plt.grid(True)
plt.show()

#optional : Evaluate using silhouette Score
# if the value is more than .7 it the best model
# if the value is less than .4 it not good at this model

from sklearn.metrics import silhouette_score
score= silhouette_score(x_scaled,clusters)
print(f'Silhouette Score:{score:.2f}')
