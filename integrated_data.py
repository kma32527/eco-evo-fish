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

with open('univ_morph.csv') as f:
    reader = csv.reader(f)
    univ_morph = list(reader)

with open('jaw_measurements.csv') as f:
    reader = csv.reader(f)
    jaw = list(reader)
univ_morph = np.array(univ_morph)
univ_morph = univ_morph[:, [0, 4, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 40, 41, 42, 43, 44, 45, 47, 48, 50, 51, 52, 53]]

placeholder = list()
for row in univ_morph:
    placeholder.append(row.tolist())
univ_morph = placeholder

raw_examples = list()
raw_examples.append(univ_morph[0] + jaw[0])
for i in range(1,len(univ_morph)):
    for j in range(1, len(jaw)):
        if univ_morph[i][0] == jaw[j][0]:
            raw_examples.append(univ_morph[i] + jaw[j])

for row in raw_examples:
    print(row)

with open('lake_jaw_morph.csv', mode='w', newline = '') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(raw_examples[0][1:len(raw_examples[0])])
    for i in range(1, len(raw_examples)):
        if raw_examples[i][0][3] == 'L':
            writer.writerow(raw_examples[i][1:len(raw_examples[0])])

with open('ls_jaw_morph.csv', mode = 'w', newline = '') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(raw_examples[0][1:len(raw_examples[0])])
    for i in range(1, len(raw_examples)):
        writer.writerow(raw_examples[i][1:len(raw_examples[0])])
