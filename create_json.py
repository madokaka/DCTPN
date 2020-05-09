import h5py
import json
import numpy as np
import time

t1 = time.time()

all = {}
all['version'] = 'VERSION 1.3'
results = {}


h5f1 = h5py.File(r"./6layer_128_validation.hdf5", 'r')
i = 0
for key in h5f1:
    i += 1
    data1 = h5f1[key]['outputs']
    key = key[2:]
    results[key] = []
    a = np.array(data1)
    cache = []
    cache_sort = []
    cache_sort_nms = []

    batch, nb_step, k = a.shape
    for batch_s in range(batch):
        for time_step in range(nb_step):
            for k_s in range(64):
                score = a[batch_s, time_step, k_s]
                start = batch_s + time_step - (k_s*2)
                end = batch_s + time_step + 1
                cache.append([start, end, score])
    cache_sort = [value for index, value in sorted(enumerate(cache), key=lambda a: a[1][2])]
    cache_sort.reverse()
    cache_sort_nms.append(cache_sort[0])
    for n in range(1, len(cache_sort)):
        biaozhi = 1
        for item in cache_sort_nms:
            intersection = max(0, min(cache_sort[n][1], item[1]) - max(cache_sort[n][0], item[0]))
            union = min(max(cache_sort[n][1], item[1]) - min(cache_sort[n][0], item[0]), cache_sort[n][1] - cache_sort[n][0] + item[1] - item[0])
            overlap = float(intersection) / (union + 1e-8)
            if overlap > 0.8:
                if cache_sort[n][2] > item[2]:
                    cache_sort_nms.remove(item)
                    biaozhi = 1
                else:
                    biaozhi = 0
        if biaozhi == 1:
            cache_sort_nms.append(cache_sort[n])
        if len(cache_sort_nms) > 99:
            break

    for item in cache_sort_nms:
        result = {}
        result['score'] = float(item[2])
        result['segment'] = [float(item[0]), float(item[1])]
        results[key].append(result)

    if i%10 == 0:
        print(i/len(h5f1))


all['results'] = results
all.update({'external_data': {'used':False}})

with open('bn_6layer_128_validation.json', 'w') as f:
    json.dump(all, f)

t2 = time.time()
print(t2 - t1)