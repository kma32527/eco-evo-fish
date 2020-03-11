import csv
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.pipeline import Pipeline
from sklearn import metrics
from scipy.stats import kde
from scipy.interpolate import griddata



with open('lake_morph.csv', 'r') as f:
    reader = csv.reader(f)
    raw_examples = list(reader)


with open('climate_data.csv', 'r') as f:
    reader = csv.reader(f)
    climate_data = list(reader)

with open('eco_site.csv', 'r') as f:
    reader = csv.reader(f)
    eco_site = list(reader)

with open('Watershed_area.csv') as f:
    reader = csv.reader(f)
    area = list(reader)

def greatest_factors(num):
    num = int(num)
    factor = 1
    for i in range(2,num):
        if num%i == 0:
            factor = i
    return factor, int(num/factor)

raw_examples = np.array(raw_examples)


lol1 = np.array(raw_examples[:, 29])
lol2 = np.array(raw_examples[:, 1:29])

raw_examples[:, 1] = lol1
for i in range(len(lol2[0])):
    raw_examples[:, i + 2] = lol2[:, i]

for j in range(len(raw_examples[0])):
    print(str(j) + '   ' + raw_examples[0][j])

body_traits = [4, 5, 6, 8, 13, 14, 16, 19]
'''
body traits:
4 dorsal fin len
5 caudal depth
6 anal fin
8 body depth
13 pect fin wid
14 pect fin len
16 pect fin area

'''

'''
shuffle = raw_examples[:, 1:len(raw_examples[0])].T
np.random.shuffle(shuffle)
raw_examples[:, 1:len(raw_examples[0])] = shuffle.T
'''
placeholder = list()
for row in raw_examples:
    placeholder.append(row.tolist())
raw_examples = placeholder


for i in range(len(raw_examples) - 1, 0, -1):
    if float(raw_examples[i][1]) < 30:
        raw_examples.pop(i)


numbers = ['1','2','3','4','5','6','7','8','9','0','.']
for j in range(len(raw_examples[0]) - 1, 0, -1):
    good = 1
    for i in range(len(raw_examples[0]) - 1, 0, -1):
        if good == 1:
            #if raw_examples[i][j] == 'NA' or raw_examples[i][j] == '' or 'raker' in raw_examples[0][j].lower():
            if not (j in body_traits
                   or 'Standard' in raw_examples[0][j]):
                for i in range(len(raw_examples)):
                    raw_examples[i].pop(j)
                good = 0
                continue
            else:
                if not (raw_examples[i][j] == '' or raw_examples[i][j] == 'NA'):
                    try:
                        raw_examples[i][j] = math.log(float("".join([c for c in str(raw_examples[i][j]) if c in numbers])))
                    except:
                        raw_examples.pop(i)
                else:
                    raw_examples.pop(i)


watersheds = list()
for i in range(1, len(raw_examples)):
    if not raw_examples[i][0] in watersheds:
        watersheds.append(raw_examples[i][0])

climate_lake_hash = {}
for j in range(1, len(climate_data[0])):
    for i in [29]:
        climate_lake_hash[climate_data[0][j].lower()[0:2]] = climate_data[i][j]

for i in range(1, len(raw_examples)):
    raw_examples[i].append(float(climate_lake_hash[row[0][0:2]]))


lake_mean = {}
lake_mse = {}
for lake in watersheds:
    lake_mean[lake] = list()
    lake_mse[lake] = list()

for j in range(2,len(raw_examples[0])):
    values = list()
    lakelist = list()
    #if j == 2:
    #    for i in range(1,len(raw_examples)):
    #        raw_examples[i][j] = (raw_examples[i][j] + raw_examples[i][j+1] + raw_examples[i][j + 2])/3
    for lake in watersheds:
        lengths = list()
        target = list()
        for i in range(1,len(raw_examples)):
            print(raw_examples[i][j])
            if raw_examples[i][0] == lake and float(raw_examples[i][j]) > 0:
                lengths.append(math.log(float(raw_examples[i][1])))
                target.append(math.log(float(raw_examples[i][j])))
        target = np.array(target)
        if len(lengths) > 0:
            new_model = LinearRegression().fit(np.array(lengths).reshape(-1,1), target)
            model_target = new_model.predict(np.array(lengths).reshape(-1,1))
            lake_mean[lake].append(float(new_model.coef_))
            lake_mse[lake].append(mean_squared_error(target, new_model.predict(np.array(lengths).reshape(-1,1))))

row, col = greatest_factors(len(lake_mean['beaver']))


x = list()
y = list()
z = list()
env1 = 29
env2 = 1

#Kennedy is an outlier lake with respects to size
lake_mean.pop('kennedy')

bad = list()

for key in lake_mean:
    #if not key in ['pachena', 'joe', 'moore']:
    #    bad.append(key)
    #else:
        some_key = key

for key in bad:
    lake_mean.pop(key)


for trait in range(len(lake_mean[some_key])):
    z.append(list())
    for lake in lake_mean:
        for j in range(1, len(climate_data[0])):
            if climate_data[0][j].lower()[0:2] == lake[0:2]:
                x.append((float(climate_data[env1][j])))
        for i in range(len(area)):
            if area[i][0].lower()[0:2] == lake[0:2]:
                y.append(math.log(float(area[i][1])))
        #        y.append(math.log(float(area[i][1])))
        #for i in range(len(eco_site)):
        #    if lake[0:2] == eco_site[i][1].lower()[0:2] and eco_site[i][2] == 'L':
        #        y.append(float(eco_site[i][29]))
        z[trait].append(math.log(float(lake_mean[lake][trait])))

z_max = list()
tot = list()
for trait in range(len(z)):
    z_max.append(np.max(z[trait]))
    tot.append(z_max[trait] - np.min(z[trait]))

# define grid.
xi = np.linspace(np.min(x),np.max(x),100)
yi = np.linspace(np.min(y),np.max(y),100)

for trait in range(len(z)):
    for i in range(len(lake_mean)):
        #plt.subplot(row, col, 1)
        #plt.plot(z[i], 0, color = [(z_max[trait] - z[trait][i])/tot[trait], 0, 0], marker = 'o')
        plt.subplot(row, col, trait + 1)
        plt.plot(x[i],y[i], color = [(z_max[trait] - z[trait][i])/tot[trait], 1 - (z_max[trait] - z[trait][i])/tot[trait], 0], marker = 'o')
        plt.title(raw_examples[0][trait + 2])
        plt.xlabel(climate_data[env1][0], fontsize=10)
        #plt.ylabel(climate_data[env2][0], fontsize=10)
        plt.ylabel('log area', fontsize = 10)
plt.show()

plt.clf()

raw_examples = np.array(raw_examples[1:len(raw_examples)])

inertia = 0
clusters = 0
for k in range (1, 30):
    if k > 1:
        prev_in = inertia
        prev_clust = clusters
    # Create a kmeans model on our data, using k clusters.  random_state helps ensure that the algorithm returns the same results each time.
    kmeans_model = KMeans(n_clusters=k, random_state=1).fit(raw_examples[:, 1:len(raw_examples[0])])
    # These are our fitted labels for clusters -- the first cluster has label 0, and the second has label 1.
    labels = kmeans_model.labels_
    clusters = kmeans_model.cluster_centers_
# Sum of distances of samples to their closest cluster center
    inertia = kmeans_model.inertia_
    print("k:",k, " log cost:", inertia)

#for k in [58]:
'''
for k in range(1, len(climate_data)):
    for i in range(len(lake_mean['beaver'])):
        for lake in watersheds:
            plt.subplot(row, col,i + 1)
            plt.title(raw_examples[0][i + 2])
            for j in range(1, len(climate_data[0])):
                if climate_data[0][j].lower()[0:2] == lake[0:2]:
                    plt.plot(float(climate_data[k][j]), 0, 'go' )
                    if lake == 'kennedy':
                        plt.plot(float(climate_data[k][j]), 0, 'ro' )
                        plt.xlabel(climate_data[k][0], fontsize = 10)
    plt.suptitle(climate_data[k][0])
    plt.show()
    plt.clf()
'''
'''
for j in range(len(lake_mean['beaver'])):
    plt.subplot(4, 4, j + 1)
    plt.title(raw_examples[0][j + 2])
    for lake in lake_mean:
        for i in range(1, len(climate_data[0])):
            if lake[0:2] == climate_data[0][i].lower()[0:2]:
                plt.plot(float(climate_data[2][i]), lake_mean[lake][j], 'rx')
'''

for j in range(2, len(raw_examples[0])):
    print(str(j - 1) + '  ' + raw_examples[0][j])


wet = list()

targetlake = 'moore'

def fit_feature(feature):
    for lake in watersheds:
        for j in range(1, len(climate_data[0])):
            if lake[0:2] == climate_data[0][j].lower()[0:2]:
                if float(climate_data[30][j]) > 1000:
                    wet.append(lake)
    print(wet)

    x_train = list()
    x_test = list()
    y_train = list()
    y_test = list()

    for i in range(1, len(raw_examples)):
        row = raw_examples[i]
        if row[0] ==targetlake:
            x_test.append(row[1:len(row)]/float(row[1]))
            if row[0] in wet:
                y_test.append(1)
            else:
                y_test.append(-1)
        else:
            x_train.append(row[1:len(row)]/float(row[1]))
            if row[0] in wet:
                y_train.append(1)
            else:
                y_train.append(-1)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

'''
for i in range(1, len(raw_examples)):
    if raw_examples[i][0] in wet:
        raw_examples[i][0] = 1
    else:
        raw_examples[i][0] = -1
'''
'''
raw_examples = np.array(raw_examples)
shuffle = raw_examples[1:len(raw_examples)]
np.random.shuffle(shuffle)
raw_examples[1:len(raw_examples)] =  shuffle
'''

'''
split = float(1/2)
x = raw_examples[1:len(raw_examples), 1:len(raw_examples[0])]
y = raw_examples[1:len(raw_examples), 0]
x_train = x[1:math.ceil(len(x)*split)]
y_train = y[1:math.ceil(len(x)*split)]
x_test = x[math.ceil(len(x)*split):len(x)]
y_test = y[math.ceil(len(x)*split):len(x)]
'''
'''
    for value in [1]:
            pclf = Pipeline([
            #        ('tfidf', TfidfTransformer()),
            #        ('norm', Normalizer()),
            ('clf',  RandomForest(n_estimators = 20, max_depth = 5)),
        ])
        pclf.fit(x_train, y_train)
        y_pred = pclf.predict(x_test)
        count = 0
        for i in range(len(y_test)):
            if y_test[i] == y_pred[i]:
                count += 1
        print(y_test - y_pred)
        print('% acc = ' + str(100 * (count/len(y_test))))
        print('size of test = ' + str(len(y_test)))
        print(metrics.classification_report(y_test, y_pred))
'''