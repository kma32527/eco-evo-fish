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

with open('lake_jaw_morph.csv') as f:
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

raw_examples = np.array(raw_examples)[:, 1:len(raw_examples[0])]



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
'''
A = 58 KT.diagonal
B = 55 KT.coupler.link (?)
C = 56 KT.input.link (?)
'''
raw_examples[0].append('KT.angle')
for i in range(1, len(raw_examples)):
    a = float(raw_examples[i][58])
    b = float(raw_examples[i][55])
    c = float(raw_examples[i][56])
    raw_examples[i].append(math.acos(((b**2 + c**2) - a**2)/(2*b*c)))

numbers = ['1','2','3','4','5','6','7','8','9','0','.']

for j in range(len(raw_examples[0]) - 1, 0, -1):
    good = 1
    for i in range(len(raw_examples) - 1, 0, -1):
        if good == 1:
            #if raw_examples[i][j] == 'NA' or raw_examples[i][j] == '' or 'raker' in raw_examples[0][j].lower():
            if not (j == 59
                    or 'Standard' in raw_examples[0][j]):
                for i in range(len(raw_examples)):
                    raw_examples[i].pop(j)
                good = 0
                continue
            else:
                try:
                    raw_examples[i][j] = float("".join([c for c in str(raw_examples[i][j]) if c in numbers]))
                except:
                    raw_examples[i][j] = -1


watersheds = list()
for i in range(1, len(raw_examples)):
    if not raw_examples[i][0] in watersheds:
        watersheds.append(raw_examples[i][0])


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
            if raw_examples[i][0] == lake and raw_examples[i][j] > 0:
                lengths.append(math.log(raw_examples[i][1]))
                target.append((raw_examples[i][j]))
        target = np.array(target)
        if len(lengths) > 0:
            #new_model = LinearRegression().fit(np.array(lengths).reshape(-1,1), target)
            #model_target = new_model.predict(np.array(lengths).reshape(-1,1))
            #lake_mean[lake].append(float(new_model.coef_))
            #lake_mse[lake].append(mean_squared_error(target, new_model.predict(np.array(lengths).reshape(-1,1))))
            xcoord = 0
            ycoord = 0
            for i in range(len(target)):
                xcoord += math.sin(target[i])
                ycoord += math.cos(target[i])
            xcoord = xcoord/len(target)
            ycoord = ycoord/len(target)
            lake_mean[lake] = math.atan2(xcoord, ycoord)

#row, col = greatest_factors(len(lake_mean['beaver']))


x = list()
y = list()
z = list()

lake_mean.pop('kennedy')

bad = list()

'''
for key in lake_mean:
    #if not key in ['pachena', 'joe', 'moore']:
    #    bad.append(key)
    #else:
    some_key = key

for key in bad:
    lake_mean.pop(key)
'''
env1 = 31
env2 = 48


for trait in [0]:
#for trait in range(len(lake_mean['beaver'])):
    for lake in lake_mean:
        for j in range(1, len(climate_data[0])):
            if climate_data[0][j].lower()[0:2] == lake[0:2]:
                x.append((float(climate_data[env1][j])))
                y.append((float(climate_data[env2][j])))
        #for i in range(len(area)):
        #    if area[i][0].lower()[0:2] == lake[0:2]:
        #        y.append(math.log(float(area[i][1])))
        #       y.append(math.log(float(area[i][1])))
        #for i in range(len(eco_site)):
        #    if lake[0:2] == eco_site[i][1].lower()[0:2] and eco_site[i][2] == 'L':
        #        y.append(float(eco_site[i][29]))
        z.append(float(lake_mean[lake]))

z_max = list()
tot = list()

z_max = np.max(z)
tot = z_max - np.min(z)

# define grid.
xi = np.linspace(np.min(x),np.max(x),100)
yi = np.linspace(np.min(y),np.max(y),100)

for trait in [0]:
#for trait in range(len(z)):
    for i in range(len(lake_mean)):
        #plt.subplot(row, col, 1)
        #plt.plot(z[i], 0, color = [(z_max[trait] - z[trait][i])/tot[trait], 0, 0], marker = 'o')
        #plt.subplot(row, col, trait + 1)
        plt.plot(x[i],y[i], color = [(z_max - z[i])/tot, 1 - (z_max - z[i])/tot, 0], marker = 'o')
        plt.title(raw_examples[0][trait + 2])
        plt.xlabel(climate_data[env1][0], fontsize=10)
        #plt.ylabel(climate_data[env2][0], fontsize=10)
        plt.ylabel(climate_data[env2][0], fontsize = 10)

print('Average KT angle')
for lake in lake_mean:
    print(lake + '  ' + str(lake_mean[lake]))
plt.show()
