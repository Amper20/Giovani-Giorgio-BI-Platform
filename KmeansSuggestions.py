import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle

if __name__ == "__main__":

    orders = pd.read_csv('archive/restaurant-1-orders.csv')
    products = pd.read_csv('archive/restaurant-1-products-price.csv')
    order_groups = orders.groupby(pd.Grouper(key='Order Number', sort=True))
    product_names = list(products["Item Name"].iloc[:])

    matrix = []
    for group_name, df_group in order_groups:
        matrix.append([0]*len(product_names))
        order_items = list(df_group["Item Name"].iloc[:])
        for order_item in order_items:
            idx = 0
            for prod in product_names:
                if(prod == order_item):
                    matrix[-1][idx] = 1
                idx += 1

    X = np.array(matrix)

    kmeans = KMeans(n_clusters=30, random_state=0).fit(X)
    centers = kmeans.cluster_centers_
    centers = np.rint(centers)
    menus = []
    for menu_sample in centers:
        menus.append([])
        for i in range(0, len(product_names)):
            if menu_sample[i] >= 0.3:
                menus[-1].append(product_names[i])
    print(menus)
    a_file = open("menus.pkl", "wb")
    pickle.dump(menus, a_file)
    a_file.close()