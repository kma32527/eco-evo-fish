import csv
import numpy as np

with open('lake_morph.csv', 'r') as f:
    reader = csv.reader(f)
    raw_examples = list(reader)

'''
0  watershed
1  dorsal.spine.1.len.mm.lateral
2  dorsal.spine.2.len.mm.lateral
3  dorsal.fin.len.mm.lateral
4  caudal.depth.mm.lateral
5  anal.fin.len.mm.lateral
6  pect.fin.insertion.len.mm.lateral
7  body.depth.mm.lateral
8  mouth.len.mm.lateral
9  snout.len.mm.lateral
10  eye.diam.mm.lateral
11  head.len.mm.lateral
12  pect.fin.wid.len.mm.lateral
13  pect.fin.len.mm.lateral
14  pect.fin.perim.mm.lateral
15  pect.fin.area.mm.lateral
16  standard.len.mm.lateral
17  buccal.cavity.length.mm
18  gape.width.mm
19  body.width.eye.mm
20  body.width.midbody.mm
21  pelvic.girdle.width.mm
22  pelvic.girdle.diamond.width.mm
23  pelvic.girdle.length.mm
24  pelvic.girdle.diamond.length.mm
25  body.width.anal.1.mm
26  body.width.anal.2.mm
27  Habitat.L.0..S.1
28  Mass.g
29  Standard.Length.mm
30  Left.Side.Pelvic.Spine.Length.mm
31  Left.Side.Plate.Count
32  Right.Side.Pelvic.Spine.Length.mm
33  Right.side.plate.count
34  Right.side.gill.raker.number.insitu
35  Right.Side.Gill.Raker.Number.dissected
36  Length.longest.raker.mm
37  Length.2nd.longest.raker.mm
38  Length.3rd.longest.raker.mm
39  raker.density.per.mm
'''

raw_examples = np.array(raw_examples)
for i in range(len(raw_examples[0])):
    raw_examples[0][i] = raw_examples[0][i].replace('.', '_')


watersheds = {}
for i in range(1, len(raw_examples)):
    if not raw_examples[i][0] in watersheds:
        watersheds[raw_examples[i][0]] = 1
    else:
        watersheds[raw_examples[i][0]] += 1

#data missing per trait
missing_trait_data = {}
for i in range(1, len(raw_examples[0])):
    counter = 0
    for j in range(len(raw_examples)):
        if raw_examples[j][i] == 'NA':
            counter += 1
    missing_trait_data[raw_examples[0][i]] = counter
with open('num_incomplete_by_trait.csv', mode='w', newline = '') as csv_file:
    writer = csv.writer(csv_file)
    for trait in missing_trait_data:
        writer.writerow([trait, str(missing_trait_data[trait])])

#data missing per lake
missing_lake_data = {}
for lake in watersheds:
    missing_lake_data[lake] = np.zeros(len(raw_examples[0]) - 1)
    for i in range(len(raw_examples)):
        for j in range(1, len(raw_examples[0])):
            if raw_examples[i][0] == lake and raw_examples[i][j] == 'NA':
                missing_lake_data[lake][j] += 1

with open('num_incomplete_by_lake.csv', mode='w', newline = '') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(np.concatenate([['watershed', 'total'],[trait for trait in missing_trait_data]]))
    for lake in missing_lake_data:
        writer.writerow(np.concatenate([[lake, str(watersheds[lake])], missing_lake_data[lake].astype('str')]))

#averages of each trait by lake

lake_mean_data = {}
numbers = ['1','2','3','4','5','6','7','8','9','0','.']
rescale = np.zeros(len(raw_examples))
for i in range(1, len(raw_examples)):
    rescale[i] = 45/float(raw_examples[i][29])
for i in range(1, len(raw_examples)):
    for j in range(1, len(raw_examples[0])):
        if not raw_examples[i][j] == 'NA' and not raw_examples[i][j] == '':
            if j == 28:
                raw_examples[i][j] = float("".join([c for c in raw_examples[i][j] if c in numbers])) * rescale[i]**3
            else:
                if not j == 29:
                    raw_examples[i][j] = float("".join([c for c in raw_examples[i][j] if c in numbers]))*rescale[i]

#31, 33
with open('lake_norm_45mm.csv', mode='w', newline = '') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(np.concatenate([raw_examples[0][0:27], raw_examples[0][28:31], [raw_examples[0][32]], raw_examples[0][34:39]]))
    for i in range(1, len(raw_examples)):
        writer.writerow(np.concatenate([raw_examples[i][0:27], raw_examples[i][28:31], [raw_examples[i][32]], raw_examples[i][34:39]]))

'''
for i in range(1, len(raw_examples)):
    rescale[i] = 45/raw_examples[i][29]
for lake in watersheds:
    lake_mean_data[lake] = np.zeros(len(raw_examples[0]) - 1)
    for j in range(1, len(raw_examples[0])):
        counter = 0
        total = 0
        for i in range(len(raw_examples)):
            if raw_examples[i][0] == lake and not raw_examples[i][j] == 'NA' and not raw_examples[i][j] == '':
                parse = "".join([c for c in raw_examples[i][j] if c in numbers])
                total += float(parse)
                counter += 1
        lake_mean_data[lake][j - 1] = total / counter

with open('watershed_lake_normalized.csv', mode='w', newline = '') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(np.concatenate([['watershed', 'total'],[trait for trait in missing_trait_data]]))
    for lake in missing_lake_data:
        writer.writerow(np.concatenate([[lake, str(watersheds[lake])], lake_mean_data[lake].astype('str')]))
'''
