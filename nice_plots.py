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

raw_examples = np.array(raw_examples)


lol1 = np.array(raw_examples[:, 29])
lol2 = np.array(raw_examples[:, 1:29])

raw_examples[:, 1] = lol1
for i in range(len(lol2[0])):
    raw_examples[:, i + 2] = lol2[:, i]


for j in range(1, len(raw_examples[0])):
    print(str(j)+ '  ' + raw_examples[0][j])


body_traits = [4, 5, 6, 8, 13, 14, 16]
'''
body traits:
4 dorsal fin len
5 caudal depth
6 anal fin
9 body depth
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
    for i in range(1, len(raw_examples)):
        if good == 1:
            if not (j in body_traits
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

count = 0
for j in range(1, len(raw_examples[0])):
    for i in range(len(raw_examples) - 1, 0, -1):
        if raw_examples[i][j] == -1:
            raw_examples.pop(i)
            count += 1
print('removed = ' + str(count))

watersheds = list()
for i in range(1, len(raw_examples)):
    if not raw_examples[i][0] in watersheds:
        watersheds.append(raw_examples[i][0])


lake_values = {}
plt.figure(1)
lake_plots = {}
i = 1
for lake in watersheds:
    lake_plots[lake] = plt.subplot(4, 4, i)
    lake_plots[lake].axes.get_xaxis().set_visible(False)
    lake_plots[lake].axes.get_yaxis().set_visible(False)
    plt.title(lake)
    i += 1

for i in range(1, len(raw_examples[0])):
    if raw_examples[0][i] == 'Standard.Length.mm':
        length_ind = i
for i in range(len(raw_examples) - 1, 0, -1):
    if raw_examples[i][length_ind] < 35:
        raw_examples.pop(i)
for j in range(1,len(raw_examples[0])):
    values = list()
    lakelist = list()
    for lake in watersheds:
        lengths = list()
        target = list()
        for i in range(1,len(raw_examples)):
            if raw_examples[i][0] == lake and raw_examples[i][j] > 0:
                lengths.append(math.log(raw_examples[i][length_ind]))
                target.append(math.log(raw_examples[i][j]))
        target = np.array(target)
        if len(lengths) > 0:
            new_model = LinearRegression().fit(np.array(lengths).reshape(-1,1), target)
            model_target = new_model.predict(np.array(lengths).reshape(-1,1))
            values.append(float(new_model.coef_))
            lake_values[lake] = float(new_model.coef_)
            lakelist.append(lake)
    ar = np.array(values).reshape(-1,1)
    val = 0
    if len(ar) > 0:
        kmeans = KMeans(n_clusters=1, random_state=0).fit(ar)
        center = kmeans.cluster_centers_
        maxi = 0
        maxind = -1
        maxmse = 0
        farthest = center
        for i in range(len(ar)):
            if np.linalg.norm(ar[i] - center) > maxmse:
                maxind = i
                maxmse = np.linalg.norm(ar[i] - center)
                maxi = np.array(ar[i])
    ar = np.array(ar).T
    i = 1
    for lake in lake_values:
        good = 1
        for i in range(1,len(raw_examples)):
            if raw_examples[i][j] == -1 and raw_examples[i][0] == lake:
                good = 0
        if good == 1:
            plt.figure(1)
            lake_plots[lake].plot(ar - center, j, 'rx')
            lake_plots[lake].plot(lake_values[lake] - center, j, 'go')


for lake in lake_plots:
    lake_plots[lake].axvline(x=0)
plt.suptitle('Comparison by Lake of Log-Linear Allometry Coefficients')
for i in range(1,len(raw_examples[0])):
    print(str(i) + '  ' + raw_examples[0][i])
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
#mng = plt.get_current_fig_manager()
#mng.full_screen_toggle()
plt.show()
