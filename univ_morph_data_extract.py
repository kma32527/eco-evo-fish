import csv
import numpy as np

with open('univ_morph.csv', 'r') as f:
    reader = csv.reader(f)
    raw_examples = list(reader)
print(raw_examples)

raw_examples = np.array(raw_examples)

'''
0   fishID.univ
3   watershed
4   habitat
6   dorsal.spine.1.len.mm.lateral
7   dorsal.spine.2.len.mm.lateral
8   dorsal.fin.len.mm.lateral
9   caudal.depth.mm.lateral
10   anal.fin.len.mm.lateral
11   pect.fin.insertion.len.mm.lateral
12   body.depth.mm.lateral
13   mouth.len.mm.lateral
14   snout.len.mm.lateral
15   eye.diam.mm.lateral
16   head.len.mm.lateral
17   pect.fin.wid.len.mm.lateral
18   pect.fin.len.mm.lateral
19   pect.fin.perim.mm.lateral
20   pect.fin.area.mm.lateral
21   standard.len.mm.lateral
26   buccal.cavity.length.mm 
27   gape.width.mm
28   body.width.eye.mm
29   body.width.midbody.mm
30   pelvic.girdle.width.mm
31   pelvic.girdle.diamond.width.mm
32   pelvic.girdle.length.mm
33   pelvic.girdle.diamond.length.mm
34   body.width.anal.1.mm
35   body.width.anal.2.mm
38   Habitat.L.0..S.1
40   Mass.g
41   Standard.Length.mm
42   Left.Side.Pelvic.Spine.Length.mm
43   Left.Side.Plate.Count
44   Right.Side.Pelvic.Spine.Length.mm
45   Right.side.plate.count
47   Right.side.gill.raker.number.insitu
48   Right.Side.Gill.Raker.Number.dissected
50   Length.longest.raker.mm
51   Length.2nd.longest.raker.mm
52   Length.3rd.longest.raker.mm
53   raker.density.per.mm
'''

raw_examples = raw_examples[:, [4, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 40, 41, 42, 43, 44, 45, 47, 48, 50, 51, 52, 53]]


with open('lake_morph.csv', mode='w', newline = '') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(raw_examples[0][1:len(raw_examples[0])])
    for i in range(1, len(raw_examples)):
        if raw_examples[i][0] == 'lake':
            writer.writerow(raw_examples[i][1:len(raw_examples[i])])

for i in range(len(raw_examples[0])):
    print(str(i) + '  ' + raw_examples[0][i])