import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

with open('lake_morph.csv', 'r') as f:
    reader = csv.reader(f)
    raw_examples = list(reader)

with open('climate_data.csv', 'r') as f:
    reader = csv.reader(f)
    climate_data = list(reader)

with open('lake_45mm_mean_by_precip.csv','r') as f:
    reader = csv.reader(f)
    precip = list(reader)

with open('lake_jaw_morph.csv') as f:
    reader = csv.reader(f)
    raw_examples = list(reader)

raw_examples = np.array(raw_examples)


lol1 = np.array(raw_examples[:, 29])
lol2 = np.array(raw_examples[:, 1:29])

raw_examples[:, 1] = lol1
for i in range(len(lol2[0])):
    raw_examples[:, i + 2] = lol2[:, i]

placeholder = list()
for row in raw_examples:
    placeholder.append(row.tolist())
raw_examples = placeholder

watersheds = list()
for i in range(1, len(precip)):
    if not precip[i][0] in watersheds:
        watersheds.append(precip[i][0])

season_precip = {}
for lake in watersheds:
    for j in range(1, len(climate_data[0])):
        if climate_data[0][j][0:2].lower() == lake[0:2]:
            season_precip[lake] = [climate_data[30][j], climate_data[31][j], climate_data[32][j], climate_data[33][j]]

lake_values = {}
plt.figure(1)
lake_plots = {}
j = 1
for lake in watersheds:
    ar = list()
    for i in range(1, len(raw_examples)):
        if raw_examples[i][0] == lake:
            plt.plot(j, math.log(float(raw_examples[i][1])), 'rx')
            ar.append(math.log(float(raw_examples[i][1])))
    print(str(j) + ' = ' + lake)
    ar = np.array(ar).reshape(-1,1)
    if len(ar) > 0:
        #kmeans = KMeans(n_clusters=2, random_state=1).fit(ar)
        #center = kmeans.cluster_centers_
        #plt.plot(j, center[0], 'go')
        #plt.plot(j, center[1], 'go')
        #plt.plot(j, center[2], 'go')
        print(lake)
        inertia = 0
        clusters = 0
        for k in range (1, 5):
            if k > 1:
                prev_in = inertia
                prev_clust = clusters
        # Create a kmeans model on our data, using k clusters.  random_state helps ensure that the algorithm returns the same results each time.
            kmeans_model = KMeans(n_clusters=k, random_state=1).fit(ar)

        # These are our fitted labels for clusters -- the first cluster has label 0, and the second has label 1.
            labels = kmeans_model.labels_
            clusters = kmeans_model.cluster_centers_
        # Sum of distances of samples to their closest cluster center
            inertia = kmeans_model.inertia_
            if k > 1:
                if prev_in - inertia < 1:
                    for point in prev_clust:
                        plt.plot(j, float(point), 'rx')
                    break

            print("k:",k, " log cost:", math.log(inertia))
    j += 1



'''
for lake in watersheds:
    lake_plots[lake] = plt.subplot(4, 4, i)
    lake_plots[lake].axes.get_xaxis().set_visible(False)
    lake_plots[lake].axes.get_yaxis().set_visible(False)
    plt.title(lake)
    i += 1
'''
plt.xticks(np.arange(1,17, step= 1))
plt.show()

###
for j in range(len(raw_examples[0])):
    print(str(j) + '  ' + raw_examples[0][j])
plt.clf()
plt.figure(1)
j = 1
for lake in watersheds:
    lake_plots[lake] = plt.subplot(4,4,j)
    plt.title(lake)
    j += 1

for i in range(len(raw_examples) - 1, 0, -1):
    if raw_examples[i][0] == 'joe' and float(raw_examples[i][29]) > 6:
        raw_examples.pop(i)

fish_length = {}
fish_mass = {}
lake_slope = {}
lake_intercept = {}
lake_points = {}
for lake in lake_plots:
    lake_slope[lake] = list()
    lake_intercept[lake] = list()
    fish_length[lake] = list()
    fish_mass[lake] = list()
    lake_points[lake] = list()
    for i in range(1, len(raw_examples)):
        for j in range(1, len(raw_examples[0])):
            if raw_examples[0][j] == 'Standard.Length.mm' and raw_examples[i][0] == lake:
                x = math.log(float(raw_examples[i][j]))
                fish_length[lake].append(x)
            if raw_examples[0][j] == 'Mass.g' and raw_examples[i][0] == lake:
                y = math.log(float(raw_examples[i][j]))
                fish_mass[lake].append(y)

    flen = np.array(fish_length[lake]).reshape(-1,1)
    fmas = np.array(fish_mass[lake])
    inertia = 0
    clusters = 0
    for k in range (1, 5):
        if k > 1:
            prev_in = inertia
            prev_clust = clusters
            prev_model = kmeans_model
        # Create a kmeans model on our data, using k clusters.  random_state helps ensure that the algorithm returns the same results each time.
        kmeans_model = KMeans(n_clusters=k, random_state=1).fit(flen, fmas)
        #scale_model = LinearRegression().fit(flen, fmas)
        # These are our fitted labels for clusters -- the first cluster has label 0, and the second has label 1.
        labels = kmeans_model.labels_
        clusters = kmeans_model.cluster_centers_
        # Sum of distances of samples to their closest cluster center
        inertia = math.log(kmeans_model.inertia_)
        if k > 1:
            if prev_in - inertia < 1000:
                lake_partition_length = list()
                lake_partition_mass = list()
                for i in range(len(prev_clust)):
                    lake_partition_length.append(list())
                    lake_partition_mass.append(list())
                print(len(lake_partition_length))
                print(len(lake_partition_mass))
                y = prev_model.predict(np.array(fish_length[lake]).reshape(-1,1))
                for i in range(len(y)):
                    for j in range(len(prev_clust)):
                        if y[i] == j:
                            lake_partition_length[j].append(float(fish_length[lake][i]))
                            lake_partition_mass[j].append(float(fish_mass[lake][i]))
                for j in range(len(lake_partition_length)):
                    new_model = LinearRegression().fit(np.array(lake_partition_length[j]).reshape(-1,1), np.array(lake_partition_mass[j]))
                    lake_slope[lake].append(new_model.coef_)
                    lake_intercept[lake].append(new_model.intercept_)
                    lake_points[lake].append(lake_partition_length[j])
                break


for lake in lake_plots:
    #for more_lakes in lake_plots:
    #    lake_plots[lake].plot(fish_length[more_lakes], fish_mass[more_lakes], 'rx')
    lake_plots[lake].plot(fish_length[lake], fish_mass[lake], 'gx')

    [min, max] = lake_plots[lake].axes.get_xlim()
    for j in range(len(lake_slope[lake])):
        x = np.linspace(math.floor(np.min(lake_points[lake][j])), math.ceil(np.max(lake_points[lake][j])), 1000)
        lake_plots[lake].plot(x, x*float(lake_slope[lake][j]) + float(lake_intercept[lake][j]), label = str(float(lake_slope[lake][j]))[0:6] )
        lake_plots[lake].legend(loc = 'lower right')
for lake in season_precip:
    print(lake[0:2]  + '   ' + str(season_precip[lake]))
plt.show()

