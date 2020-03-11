import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


split_type = 'sex'
lake = 'muchalat'

with open('ls_jaw_morph.csv') as f:
    reader = csv.reader(f)
    raw_examples = list(reader)

with open('climate_data.csv', 'r') as f:
    reader = csv.reader(f)
    climate_data = list(reader)

raw_examples[0].append('KT.angle')
for i in range(1, len(raw_examples)):
    a = float(raw_examples[i][59])
    b = float(raw_examples[i][56])
    c = float(raw_examples[i][57])
    raw_examples[i].append(math.acos(((b**2 + c**2) - a**2)/(2*b*c)))

def greatest_factors(num):
    num = int(num)
    factor = 1
    for i in range(2,num):
        if num%i == 0:
            factor = i
    if factor == 1:
        return greatest_factors(num + 1)
    return factor, int(num/factor)

#move length to 1
raw_examples = np.array(raw_examples)[:, 1:len(raw_examples[0])]
lol1 = np.array(raw_examples[:, 29])
lol2 = np.array(raw_examples[:, 1:29])
raw_examples[:, 1] = lol1
for i in range(len(lol2[0])):
    raw_examples[:, i + 2] = lol2[:, i]




'''
shuffle = raw_examples[:, 1:len(raw_examples[0])].T
np.random.shuffle(shuffle)
raw_examples[:, 1:len(raw_examples[0])] = shuffle.T
'''
placeholder = list()
for row in raw_examples:
    placeholder.append(row.tolist())
raw_examples = placeholder


for j in range(len(raw_examples[0])):
    print(str(j) + '  ' + raw_examples[0][j])


for i in range(1, len(raw_examples)):
    if 'm' in raw_examples[i][44].lower():
        raw_examples[i][44] = 1
    elif 'f' in raw_examples[i][44].lower():
        raw_examples[i][44] = -1
    else:
        raw_examples[i][44] = 0
    if 'l' in raw_examples[i][42].lower():
        raw_examples[i][42] = 1
    elif 's' in raw_examples[i][42].lower():
        raw_examples[i][42] = -1


for i in range(len(raw_examples) - 1, 0, -1):
    if not lake in raw_examples[i][0]:
    #if not raw_examples[i][44] == -1:
        print(raw_examples[i][0])
        raw_examples.pop(i)
body_traits = [4, 5, 6, 8, 13, 14, 16]
numbers = ['1','2','3','4','5','6','7','8','9','0','.']
for j in range(len(raw_examples[0]) - 1, 0, -1):
    good = 1
    for i in range(len(raw_examples) - 1, 0, -1):
        if good == 1:
            #if len(numbers) < 0:
            if not (j in body_traits
            #if not('raker.density' in raw_examples[0][j] or 'longest.raker' in raw_examples[0][j]
                    or raw_examples[0][j] == 'habitat' or raw_examples[0][j] == 'sex' or 'Left.Side.Plate' in raw_examples[0][j]
                    or 'Standard' in raw_examples[0][j]):
                for i in range(len(raw_examples)):
                    raw_examples[i].pop(j)
                good = 0
                i = 0
            else:
                if not raw_examples[i][j] == 'NA' and not raw_examples[i][j] == '':
                    if not ('Left.Side.Plate' in raw_examples[0][j] or 'sex' in raw_examples[0][j] or raw_examples[0][j] == 'habitat'):
                        print(raw_examples[i][j])
                        raw_examples[i][j] = math.log10(float("".join([c for c in str(raw_examples[i][j]) if c in numbers])))
                    elif 'Left.Side.Plate' in raw_examples[0][j]:
                        raw_examples[i][j] = float("".join([c for c in str(raw_examples[i][j]) if c in numbers]))
                else:
                    raw_examples.pop(i)

for j in range(1, len(raw_examples[0])):
    for i in range(len(raw_examples) - 1, 0, -1):
        if raw_examples[i][j] == -1 and not 'sex' in raw_examples[0][j] and not 'habitat' in raw_examples[0][j]:
            raw_examples.pop(i)

watersheds = list()
for i in range(1, len(raw_examples)):
    if not raw_examples[i][0] in watersheds:
        watersheds.append(raw_examples[i][0])

plt.figure(1)
plt.title(lake + ' features')
feature_plots = {}
i = 1
num_feat = len(raw_examples[0]) - 1
#factor1, factor2 = greatest_factors(num_feat - 1)
factor2, factor1 = greatest_factors(num_feat - 1)
'''
for j in range(2, len(raw_examples[0])):
    feature_plots[raw_examples[0][j]] = plt.subplot(factor1, factor2, j)
    #plt.xlabel('Log Length (mm)')
    #plt.ylabel('Log Gape Width (mm)')
    #feature_plots[raw_examples[0][j]].axes.get_xaxis().set_visible(False)
    #feature_plots[raw_examples[0][j]].axes.get_yaxis().set_visible(False)
    for i in range(1, len(raw_examples)):
        plt.plot(raw_examples[i][1], raw_examples[i][j], 'rx')
    plt.title(raw_examples[0][j])
    j += 1
'''

for j in range(2, len(raw_examples[0])):
    feature_plots[raw_examples[0][j]] = plt.subplot(factor1, factor2, j - 1)
    #feature_plots[raw_examples[0][j]] = plt.subplot(3, 4, j-1)
    plt.title(raw_examples[0][j])

for j in range(1, len(raw_examples[0])):
    if raw_examples[0][j] == 'sex':
        plateind = j
    splitval = 0

for i in range(1, len(raw_examples)):
        if float(raw_examples[i][plateind]) > splitval:
            marker = 'bx'
        elif raw_examples[i][plateind] < splitval:
            marker = 'rx'
        else:
            marker = 'gx'
        for j in range(2, len(raw_examples[0])):
            feature_plots[raw_examples[0][j]].plot(raw_examples[i][1], raw_examples[i][j], marker)

examples = np.array(raw_examples)[1:len(raw_examples), 1:len(raw_examples[0])].astype('float')
lengths = examples[:, 0].reshape(-1, 1)
mse = list()
for j in range(2,len(raw_examples[0])):
    target = examples[:, j-1]
    if len(lengths) > 0:
        new_model = LinearRegression().fit(lengths, target)
        model_target = new_model.predict(lengths)
        [min, max] = feature_plots[raw_examples[0][j]].axes.get_xlim()
        x = np.linspace(min, max, 1000)
        #mse.append(mean_squared_error(model_target, target))
        avg = 0
        for i in range(len(model_target)):
            avg += abs(model_target[i] - target[i])
        avg = avg/len(model_target)
        mse.append(avg)
        slope = float(new_model.coef_)
        intercept = float(new_model.intercept_)
        try:
            feature_plots[raw_examples[0][j]].plot(x, x*slope + intercept, label = 'MSE = ' + str(mse[j-2])[0:8])
        except:
            feature_plots[raw_examples[0][j]].plot(x, x*slope + intercept, label = 'MSE = ' + str(mse[j-2]))

for i in range(1, len(raw_examples)):
    if not float(raw_examples[i][plateind]) == splitval:
        if raw_examples[i][plateind] > splitval:
            marker = 'bx'
        else:
            marker = 'rx'
    else:
        marker = 'gx'
    for j in range(2, len(raw_examples[0])):
        feature_plots[raw_examples[0][j]].plot(raw_examples[i][1], raw_examples[i][j], marker)
#for j in range(2, len(raw_examples[0])):
#    for i in range(1, len(raw_examples)):
#        feature_plots[raw_examples[0][j]].plot(raw_examples[i][1], raw_examples[i][j], 'rx')
    #feature_plots[raw_examples[0][j]].legend(loc = 'upper left')
#    plt.title('Village Bay Stickleback Gape Width')
for j in range(1, len(raw_examples[0])):
    print(str(j) + '  ' + raw_examples[0][j])
plt.suptitle(lake)
plt.show()