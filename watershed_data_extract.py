import csv
import numpy as np
import math

with open('climate_data.csv', 'r') as f:
    reader = csv.reader(f)
    raw_examples = list(reader)

'''
0  Site
1  Elev
2  MAT - mean annual temp
3  MWMT - mean warmest month temp
4  MCMT - mean coldest month temp
5  TD - temp diff between MWMT and MCMT
6  MAP - mean annual precipitation
7  MSP - mean annual summer precipitation
8  AHM - annual heat-moisture index
9  SHM - summer heat-moisture index
10  DD_0 - deg days below 0
11  DD5 - deg days above 5
12  DD_18 - deg days below 18 
13  DD18 - deg days above 18 
14  NFFD - # frost free days
15  bFFP - start of FFP
16  eFFP - end of FFP
17  FFP - frost free period
18  Tmax_wt 
19  Tmax_sp
20  Tmax_sm
21  Tmax_at
22  Tmin_wt
23  Tmin_sp
24  Tmin_sm
25  Tmin_at
26  Tave_wt
27  Tave_sp
28  Tave_sm
29  Tave_at
30  PPT_wt
31  PPT_sp
32  PPT_sm
33  PPT_at
34  Rad_wt
35  Rad_sp
36  Rad_sm
37  Rad_at
38  DD_0_wt
39  DD_0_sp
40  DD_0_sm
41  DD_0_at
42  DD5_wt
43  DD5_sp
44  DD5_sm
45  DD5_at
46  DD_18_wt
47  DD_18_sp
48  DD_18_sm
49  DD_18_at
50  DD18_wt
51  DD18_sp
52  DD18_sm
53  DD18_at
54  NFFD_wt
55  NFFD_sp
56  NFFD_sm
57  NFFD_at
58  PAS_wt
59  PAS_sp
60  PAS_sm
61  PAS_at
62  Eref_wt
63  Eref_sp
64  Eref_sm
65  Eref_at
'''

dry_lakes = list()
wet_lakes = list()
lakes = {}
for trait in [6, 30, 31, 32, 33]:
    for i in range(1, len(raw_examples[0])):
        lakes[raw_examples[0][i]] = float(raw_examples[trait][i])
    print(raw_examples[trait][0])
    while len(lakes) > 0:
        mini = min(lakes, key=lakes.get)
        print(mini + '         ' + str(lakes.pop(mini)))


def find_coeff_var():
    with open('climate_data.csv', 'r') as f:
        reader = csv.reader(f)
        raw_examples = list(reader)

    coeff = list()
    for i in range(1, len(raw_examples)):
        mean = 0
        variance = 0
        for j in range(1, len(raw_examples[0])):
            mean += float(raw_examples[i][j])
        mean /= (len(raw_examples[0]) - 1)
        for j in range(1, len(raw_examples[0])):
            variance += (float(raw_examples[i][j]) - mean)**2
        c = str(-1)
        if not mean == 0:
            c = str(math.sqrt(variance)/mean)
        print(raw_examples[i][0] + '   ' + c)
        coeff.append(c)
    with open('climate_coeff_var.csv', mode='w', newline = '') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Trait', 'Coefficient'])
        for i in range(len(coeff)):
            writer.writerow([raw_examples[i + 1][0], coeff[i]])

'''
for i in range(len(raw_examples)):
    print(str(i) + '  ' + raw_examples[i][0])
'''