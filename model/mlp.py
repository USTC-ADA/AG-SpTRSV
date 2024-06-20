import csv
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.optim import Adam

import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import PolynomialFeatures

import pickle

import mlp_model_def

if (len(sys.argv) != 5):
    print("Usage: python mlp.py {perf result} {matrix info} {output name} {train_prop}")
    exit(-1)

perf_file = sys.argv[1]
matrix_info_file = sys.argv[2]
output_name = sys.argv[3]
train_prop = float(sys.argv[4])

matrix_data = list(csv.reader(open(matrix_info_file)))
perf_data = list(csv.reader(open(perf_file)))

matrix_table_head = ['matrix_name', 'm', 'nnz', 'avg_rnnz', 'max_rnnz', \
'cov_rnnz', 'avg_lnnz', 'max_lnnz', 'cov_lnnz', 'dep_dist', 'reverse', 'layer_num']
perf_table_head = ['matrix', 'tbs', 'sws', 'rp', 'alpha', 'lp', 'ls', 'ws', 'rg', 'time']

mt_matrix_name_idx = matrix_table_head.index('matrix_name')
mt_m_idx = matrix_table_head.index('m')
mt_nnz_idx = matrix_table_head.index('nnz')
mt_avg_nnz_idx = matrix_table_head.index('avg_rnnz')
mt_csr_coe_idx = matrix_table_head.index('cov_rnnz')
mt_avg_layer_idx = matrix_table_head.index('avg_lnnz')
mt_lnum_coe_idx = matrix_table_head.index('cov_lnnz')
mt_dep_dist_idx = matrix_table_head.index('dep_dist')
mt_reverse_idx = matrix_table_head.index('reverse')

pf_matrix_name_idx = perf_table_head.index('matrix')

# search space configuration
pf_idx = []
pf_space = []

pf_idx.append(perf_table_head.index('tbs'))
pf_space.append([64, 256, 1024])
pf_idx.append(perf_table_head.index('sws'))
pf_space.append([1, 4, 8])
pf_idx.append(perf_table_head.index('rp'))
pf_space.append([0, 1, 2])
pf_idx.append(perf_table_head.index('alpha'))
pf_space.append([1, 4, 8, 16, 32])
pf_idx.append(perf_table_head.index('lp'))
pf_space.append([0, 1])
pf_idx.append(perf_table_head.index('ws'))
pf_space.append([0, 2])
pf_idx.append(perf_table_head.index('rg'))
pf_space.append([0, 1])

final_subset = \
[
    'tmt_sym', 'cant', 'chipcool0', 'delaunay_n23', 'atmosmodd', 'nlpkkt80', 
    'europe_osm', 'hugetrace-00000', 'hugebubbles-00000', 'road_central', 'road_usa',
    'kron_g500-logn21', 'wiki-Talk', 'arabic-2005', 'FullChip', 'ASIC_680ks', 'bundle_adj', 'lp1', 'c-big','circuit5M'
]

le = LabelEncoder()
perf_set = list(filter(lambda x: '08blocks' == x[pf_matrix_name_idx], perf_data))
search_space = []
for perf_item in perf_set:
    perf_item_new = perf_item[1:-1]
    search_space.append(list([int(d) for d in perf_item_new]))
space_len = len(search_space)

pf_time_idx = perf_table_head.index('time')

perf_data = perf_data[1:]

x_data = []
y_gflops = []
y_label = []

perf_index = 0
while (perf_index < len(perf_data)):

    perf_item = perf_data[perf_index]
    matrix_name = perf_item[pf_matrix_name_idx]

    should_unseen = 0
    # unseen
    for unseen in final_subset:
        if unseen == matrix_name:
            should_unseen = 0

    perf_set = perf_data[perf_index: min(perf_index + space_len, len(perf_data))]
    perf_index += len(perf_set)

    if should_unseen:
        print(matrix_name, "continue")
        continue

    if (len(perf_set) != space_len):
        continue

    matrix_name = perf_item[pf_matrix_name_idx]

    matrix_info = list(filter(lambda x: x[pf_matrix_name_idx] == matrix_name, matrix_data))
    if (len(matrix_info) == 0):
        continue
    matrix_info = matrix_info[0]

    nnz = float(matrix_info[mt_nnz_idx])
    m = float(matrix_info[mt_m_idx])

    if (nnz < 1e3):
        continue

    info_item = [float(item) for item in matrix_info[1:]]
    info_item.pop(-2)
    x_data.append(info_item)

    y_item = []
    for perf_item in perf_set:
        time_item = float(perf_item[pf_time_idx])
        gflops = nnz / time_item
        y_item.append(gflops)
    y_max = max(y_item)
    y_min = 0
    
    y_item = [[(1.0 if item / y_max >= 0.85 else 0.0) for item in y_item], y_item]

    y_label.append(y_item)

poly = PolynomialFeatures(mlp_model_def.polynomial_n)
x_data = poly.fit_transform(x_data)
scaler = StandardScaler().fit(x_data)
x_data = scaler.transform(x_data)

x_data = np.array(x_data, dtype=np.float32)

y_label = np.array(y_label, dtype=np.float32)

x_data = torch.tensor(x_data)
y_label = torch.tensor(y_label)

dataset = Data.TensorDataset(x_data, y_label)

train_size = int(len(dataset) * train_prop)
test_size = len(dataset) - train_size

train_dataset, test_dataset = Data.random_split(dataset, [train_size, test_size])

batch_size = 1

# 把dataset放入DataLoader
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, \
    shuffle=False, num_workers=2)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, \
    shuffle=False, num_workers=2)

model = mlp_model_def.MLP(x_data[0].size(0), 64, 64, 64, space_len, search_space)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

model = model.to(device)

model.train()

print_len = 100

for epoch in range(50):
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        x, y = batch

        optimizer.zero_grad()

        res = model(x.to(device))

        loss = F.binary_cross_entropy(res, (y.to(device))[:,0,:])

        running_loss += loss.item()

        loss.backward()
        optimizer.step()

        if (i + 1) % print_len == 0:
            print('[%d, %5d] loss: %.3f' % (epoch, i, running_loss / print_len / batch_size))
            running_loss = 0.0

    print(f"epoch {epoch} done!")

model.eval()

avg_loss = 0.0
avg_correct = 0.0
avg_mape = 0.0
mape_count = 0

for i, batch in enumerate(test_loader):
    x, y = batch

    y_pred = model(x.to(device))
    pred_max = torch.argmax(y_pred, dim=1)

    for i, item in enumerate(y):
        if (y[i][0][pred_max[i]] == 1):
            avg_correct += 1
        
        max_gflops = max(y[i][1][:])
        ag_gflops = y[i][1][pred_max[i]]

        if (max_gflops):
            avg_mape += ag_gflops / max_gflops
            mape_count += 1
    
    avg_loss += F.binary_cross_entropy(y_pred, y[:,0,:].to(device))


if (test_size):
    avg_loss /= test_size
    avg_correct /= test_size
    avg_mape /= mape_count
    print("los: ", avg_loss)
    print("accuracy: ",  avg_correct)
    print("avg_mape: ", avg_mape)

#subset test

avg_mape = 0.0
#f = open('mlp_type_result.csv', 'w')

for matrix_item in final_subset:

    matrix_name = matrix_item
    perf_name = matrix_item

    perf_set = list(filter(lambda x: x[pf_matrix_name_idx] == perf_name, perf_data))
    matrix_info = list(filter(lambda x: x[pf_matrix_name_idx] == matrix_name, matrix_data))[0]

    matrix_info = [float(item) for item in matrix_info[1:]]
    matrix_info.pop(-2)
    m = float(matrix_info[mt_m_idx])
    nnz = float(matrix_info[mt_nnz_idx])
    matrix_info = np.array(matrix_info, dtype=np.float32)
    matrix_info = torch.tensor(matrix_info).reshape(1, -1)

    x_data = poly.fit_transform(matrix_info)
    x_data = scaler.transform(x_data)

    x_data = torch.tensor(x_data, dtype = torch.float32)

    time_set = [float(item[pf_time_idx]) for item in perf_set]
    gflops = [(2 * nnz + m) / float(item[pf_time_idx]) for item in perf_set]

    y_pred = model(x_data.to(device))
    pred_max = torch.argmax(y_pred, dim=1)

    print("matrix name: ", matrix_name)
    print("best gflops: {:.2f}".format(max(gflops)))
    print("pred gflops: {:.2f}".format(gflops[pred_max]))
    print(" ")

    #f.write("{},{:.2f},{:.2f}\n".format(matrix_name, min(time_set), time_set[pred_max]))

    avg_mape += gflops[pred_max] / max(gflops)

#f.close()

print("subset mape: ", avg_mape / len(final_subset))

model.scaler = scaler

with open(output_name, 'wb') as f:
    pickle.dump(model, f)