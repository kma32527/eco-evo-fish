import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import umap.umap_ as umap
import time

#tmax_wt
#habitat

lake = None
include_lake = True
include_stream = True
include_male = True
include_female = True
include_climate = True
inclusion_list = ['tmax_wt']
removed_lakes = ['kennedy']
#ppt, watersheds, sex, habitat, elevation
mode = 'split'
split_type = 'habitat'
double_split = 'tmax_wt'
split_val = 0
double_split_val = 4
keep_high = True
keep_low = True
triple_split = ''


with open('ls_jaw_morph.csv', 'r') as f:
    reader = csv.reader(f)
    raw_examples = list(reader)

with open('climate_data.csv', 'r') as f:
    reader = csv.reader(f)
    climate_data = list(reader)

with open('Watershed_area.csv', 'r') as f:
    reader = csv.reader(f)
    watershed_areas = list(reader)

def greatest_factors(num):
    num = int(num)
    factor = 1
    for i in range(2,num):
        if num%i == 0:
            factor = i
    if factor == 1:
        return greatest_factors(num + 1)
    return factor, int(num/factor)

for i in range(len(raw_examples) - 1, 0, -1):
    if raw_examples[i][1] in removed_lakes:
        #print(raw_examples[i][1])
        raw_examples.pop(i)
'''
raw_examples[0].append('KT.angle')
for i in range(1, len(raw_examples)):
    a = float(raw_examples[i][59])
    b = float(raw_examples[i][56])
    c = float(raw_examples[i][57])
    raw_examples[i].append(math.acos(((b**2 + c**2) - a**2)/(2*b*c)))
'''
#move length to 1
raw_examples = np.array(raw_examples)[:, 1:len(raw_examples[0])]
lol1 = np.array(raw_examples[:, 29])
lol2 = np.array(raw_examples[:, 1:29])
raw_examples[:, 1] = lol1
for i in range(len(lol2[0])):
    raw_examples[:, i + 2] = lol2[:, i]

shuffle = raw_examples[1:len(raw_examples), :]
np.random.shuffle(shuffle)
raw_examples[1:len(raw_examples), :] = shuffle


placeholder = list()
for row in raw_examples:
    placeholder.append(row.tolist())
raw_examples = placeholder

temp_things = [2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

climate_labels = list()
if include_climate == True:
    for i in range(1, len(climate_data)):
        raw_examples[0].append(climate_data[i][0].lower())
        if not climate_data[i][0].lower() == split_type and not climate_data[i][0].lower() == double_split:
            climate_labels.append(climate_data[i][0].lower())
        to_kelvin = 0
        if i in temp_things:
            to_kelvin = 273
        for things in raw_examples:
            for j in range(1, len(climate_data[0])):
                if things[0][0:2] == climate_data[0][j].lower()[0:2]:
                    things.append(float(climate_data[i][j]))

'''
raw_examples[0].append('areas')
for row in raw_examples:
    for j in range(1, len(watershed_areas)):
        if row[0][0:2] == watershed_areas[j][0][0:2]:
            row.append(watershed_areas[j][1])
'''
for j in range(len(raw_examples[0])):
    print(str(j) + '  ' + raw_examples[0][j])


for i in range(1, len(raw_examples)):
    if 'm' in raw_examples[i][44].lower():
        raw_examples[i][44] = 1
    elif 'f' in raw_examples[i][44].lower():
        raw_examples[i][44] = -1
    else:
        raw_examples[i][44] = 0
    if 'l' in raw_examples[i][42].lower()[0]:
        raw_examples[i][42] = 1
    elif 's' in raw_examples[i][42].lower():
        raw_examples[i][42] = -1

for i in range(len(raw_examples) - 1, 0, -1):
    if (not include_lake == True and raw_examples[i][42] == 1)\
            or (not include_stream == True and raw_examples[i][42] == -1)\
            or (not include_male == True and raw_examples[i][44] == 1)\
            or (not include_female == True and raw_examples[i][44] ==-1)\
            or not (lake == None or raw_examples[i][0] == lake):
            #print(raw_examples[i][0])
            raw_examples.pop(i)

watersheds = list()
for i in range(1, len(raw_examples)):
    if not raw_examples[i][0] in watersheds:
        watersheds.append(raw_examples[i][0])

for j in range(1,len(raw_examples[0])):
    print(str(j) + ' ' + raw_examples[0][j])

head_traits = [9, 10, 11, 12, 19, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
body_traits = [4, 5, 6, 8, 13, 14, 16]
numbers = ['1','2','3','4','5','6','7','8','9','0','.']
for j in range(len(raw_examples[0]) - 1, -1, -1):
    good = 1
    for i in range(len(raw_examples) - 1, 0, -1):
        if good == 1:
            if ('Plate' in raw_examples[0][j] or raw_examples[0][j] in climate_labels or 'fish' in raw_examples[0][j] or 'longest.raker' in raw_examples[0][j]):
                for i in range(len(raw_examples)):
                    raw_examples[i].pop(j)
                good = 0
                i = 0
            else:
                try:
                    if not raw_examples[i][j] == 'NA' and not raw_examples[i][j] == '':
                        if not ('Left.Side.Plate' in raw_examples[0][j] or 'sex' in raw_examples[0][j] or raw_examples[0][j] == 'habitat' or raw_examples[0][j] == split_type or raw_examples[0][j] == double_split or raw_examples[0][j] in climate_labels):
                            raw_examples[i][j] = math.log10(float("".join([c for c in str(raw_examples[i][j]) if c in numbers])))
                        elif raw_examples[0][j] == 'sex' or raw_examples[0][j] == 'habitat' or raw_examples[0][j] in climate_labels or raw_examples[0][j] == split_type:
                            raw_examples[i][j] = raw_examples[i][j]
                        else:
                            raw_examples[i][j] = float("".join([c for c in str(raw_examples[i][j]) if c in numbers]))
                    else:
                        raw_examples.pop(i)
                except:
                    print('preprocess error on ' + raw_examples[0][j])
                    for i in range(len(raw_examples)):
                        raw_examples[i].pop(j)
                    good = 0
                    i = 0
for thing in raw_examples[0]:
    for thing2 in thing:
        if thing2 in climate_labels:
            print('sad')


print(str(j) + '   ' + raw_examples[0][j])
watershed_means = {}
watershed_variances = {}
feature_plots = {}

def split_plot(split_type, split_val):
    target_ind = 0
    for j in range(1, len(raw_examples[0])):
        if raw_examples[0][j] == split_type:
            target_ind = j
    double_ind = 0
    for j in range(1, len(raw_examples[0])):
        if raw_examples[0][j] == double_split:
            double_ind = j
    row, col = greatest_factors(len(raw_examples[0]))
    for j in range(1, len(raw_examples[0])):
        feature_plots[raw_examples[0][j]] = plt.subplot(row,col,j)
        plt.title(raw_examples[0][j])
        mlengths = list()
        flengths = list()
        mtargets = list()
        ftargets = list()
        for i in range(1, len(raw_examples)):
            first = 0
            second = 0
            third = 0
            if not double_ind == 0 and raw_examples[i][double_ind] > 0:
                third = 1
            if float(raw_examples[i][target_ind]) > split_val:
                if keep_high == False:
                    continue
                mlengths.append(raw_examples[i][1])
                mtargets.append(raw_examples[i][j])
                first = 1
            else:
                if keep_low ==False:
                    continue
                flengths.append(raw_examples[i][1])
                ftargets.append(raw_examples[i][j])
                point_colour = 'red'
                second = 1
            plt.plot(raw_examples[i][1], raw_examples[i][j], color = [first, second, third], marker = 'x', fillstyle = 'none')
        mlengths = np.array(mlengths).reshape(-1, 1)
        flengths = np.array(flengths).reshape(-1, 1)
        mtargets = np.array(mtargets)
        ftargets = np.array(ftargets)
        [min, max] = feature_plots[raw_examples[0][j]].axes.get_xlim()
        x = np.linspace(min, max, 1000)
        #try:
        #    m_model = LinearRegression().fit(mlengths, mtargets)
        #    plt.plot(x, x*(m_model.coef_) + m_model.intercept_, color = 'blue')
        #except:
        #    print('Error on: ' + lake + ' male')
        #try:
        #    f_model = LinearRegression().fit(flengths, ftargets)
        #    plt.plot(x, x*(f_model.coef_) + f_model.intercept_, color = 'red')
        #except:
        #    print('Error on: ' + lake + ' female')
    plt.show()

def construct_visual(raw_examples):
    for j in range(len(raw_examples[0])):
        print(str(j) + ' ' + raw_examples[0][j])    
    target_ind = 0
    double_ind = 0
    for j in range(1, len(raw_examples[0])):
        if raw_examples[0][j] == split_type:
            target_ind = j
        if raw_examples[0][j] == double_split:
            double_ind = j
    targets = list()
    for i in range(1, len(raw_examples)):
        double = 0
        #lake = 1, stream = 0
        if float(raw_examples[i][double_ind]) > double_split_val:
            double = 1

        #hot = red, cold = green
        if float(raw_examples[i][target_ind]) > split_val:
            targets.append([1,0,double])
            #targets.append(1)
        else:
            targets.append([0,1,double])
            #targets.append(0)
        if mode=='split':
            raw_examples[i][double_ind]=int(double)
    print(targets)
    
    raw_examples = np.array(raw_examples)
    data = list()

    for i in range(1, len(raw_examples[0])):
       #if not i == double_ind:
           data.append(raw_examples[1:len(raw_examples), i].tolist())
    data = np.array(data).T
    plt.clf()

    '''
    n_components = 5 for habitat
    '''

    i = 1
    d = 0.45
    n = 10
   #for d in [0.45]: #best for tmax_wt
   #for d in [0]:
    for d in [.1, .2, .3, .4, .5, .6]:
    #for n in [2, 5, 10, 15, 20, 30]:
        plt.subplot(2,3,i)
        standard_embedding = umap.UMAP(n_neighbors = 30, min_dist = d, n_components = 2, init = 'random', metric = 'correlation').fit_transform(data)
        plt.title(split_type + ' d = ' +str(d))
        plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c = targets, s=1)
        i+=1
    plt.show()


#split_plot(split_type = split_type, split_val = split_val)
construct_visual(raw_examples)
