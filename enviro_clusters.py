import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

with open('climate_data.csv', 'r') as f:
    reader = csv.reader(f)
    climate_data = list(reader)


for j in range(1, len(climate_data)):
    for i in range(1, len(climate_data[0])):
        plt.plot(float(climate_data[j][i]), 0, 'rx')
    plt.title(climate_data[j][0])
    plt.show()
    plt.clf()
