import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

if __name__ == "__main__":

    restaurant_training_set = pd.read_csv('archive/restaurant-1-orders.csv')
    training_set = restaurant_training_set.groupby(pd.Grouper(key='Order Number', sort=True))

    X = np.array([[1, 2], [1, 4], [1, 0],
               [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    kmeans.predict([[0, 0], [12, 3]])