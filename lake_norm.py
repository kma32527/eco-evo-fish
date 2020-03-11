import csv
import numpy as np
import math

with open('lake_45mm_mean.csv', 'r') as f:
    reader = csv.reader(f)
    raw_examples = list(reader)

'''
0  watershed
1  dorsal_spine_1_len_mm_lateral
2  dorsal_spine_2_len_mm_lateral
3  dorsal_fin_len_mm_lateral
4  caudal_depth_mm_lateral
5  anal_fin_len_mm_lateral
6  pect_fin_insertion_len_mm_lateral
7  body_depth_mm_lateral
8  mouth_len_mm_lateral
9  snout_len_mm_lateral
10  eye_diam_mm_lateral
11  head_len_mm_lateral
12  pect_fin_wid_len_mm_lateral
13  pect_fin_len_mm_lateral
14  pect_fin_perim_mm_lateral
15  pect_fin_area_mm_lateral
16  standard_len_mm_lateral
17  buccal_cavity_length_mm
18  gape_width_mm
19  body_width_eye_mm
20  body_width_midbody_mm
21  pelvic_girdle_width_mm
22  pelvic_girdle_diamond_width_mm
23  pelvic_girdle_length_mm
24  pelvic_girdle_diamond_length_mm
25  body_width_anal_1_mm
26  body_width_anal_2_mm
27  Mass_g
28  Standard_Length_mm
29  Left_Side_Pelvic_Spine_Length_mm
30  Right_Side_Pelvic_Spine_Length_mm
31  Right_side_gill_raker_number_insitu
32  Right_Side_Gill_Raker_Number_dissected
33  Length_longest_raker_mm
34  Length_2nd_longest_raker_mm
35  Length_3rd_longest_raker_mm
'''

def find_coeff_var():
    coeff = list()
    for j in range(2, len(raw_examples[0])):
        mean = 0
        variance = 0
        for i in range(1, len(raw_examples)):
            mean += float(raw_examples[i][j])
        mean /= (len(raw_examples) - 1)
        for i in range(1, len(raw_examples)):
            variance += (float(raw_examples[i][j]) - mean)**2
        print(raw_examples[0][j] + '   ' + str(math.sqrt(variance)/mean))
        coeff.append(str(math.sqrt(variance)/mean))
    with open('lake_coeff_var.csv', mode='w', newline = '') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Trait', 'Coefficient'])
        for i in range(len(coeff)):
            writer.writerow([raw_examples[0][i + 2], coeff[i]])

def find_means():
    with open('lake_norm_45mm.csv', 'r') as f:
        reader = csv.reader(f)
        raw_examples = list(reader)
    watersheds = {}
    for i in range(1, len(raw_examples)):
        if not raw_examples[i][0] in watersheds:
            watersheds[raw_examples[i][0]] = 1
        else:
            watersheds[raw_examples[i][0]] += 1

    missing_trait_data = {}
    for i in range(1, len(raw_examples[0])):
        counter = 0
        for j in range(len(raw_examples)):
            if raw_examples[j][i] == 'NA':
                counter += 1
        missing_trait_data[raw_examples[0][i]] = counter

    lake_mean_data = {}
    numbers = ['1','2','3','4','5','6','7','8','9','0','.']
    for lake in watersheds:
        lake_mean_data[lake] = np.zeros(len(raw_examples[0]) - 1)
        for j in range(1, len(raw_examples[0])):
            counter = 0
            total = 0
            for i in range(len(raw_examples)):
                if raw_examples[i][0] == lake and not 'NA' in raw_examples[i] and not '' in raw_examples[i]:
                    parse = "".join([c for c in raw_examples[i][j] if c in numbers])
                    total += float(parse)
                    counter += 1
            lake_mean_data[lake][j - 1] = total / max(counter, 1)

    with open('lake_45mm_mean.csv', mode='w', newline = '') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(np.concatenate([['watershed', 'total'],[trait for trait in missing_trait_data]]))
        for lake in lake_mean_data:
            if not lake == 'pye':
                writer.writerow(np.concatenate([[lake, str(watersheds[lake])], lake_mean_data[lake].astype('str')]))


'''
for i in range(len(raw_examples[0])):
    print(str(i) + '  ' + raw_examples[0][i])
'''