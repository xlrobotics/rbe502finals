import csv, os
import datetime
import matplotlib.pyplot as plt
import numpy as np
from gpar import GPARRegressor
import pandas as pd
from lab import B

# B.epsilon = 1e-8
#
#
# def date_to_decimal_year(date):
#     start = datetime.date(date.year, 1, 1).toordinal()
#     year_length = datetime.date(date.year + 1, 1, 1).toordinal() - start
#     return date.year + float(date.toordinal() - start) / year_length
#
#
# def safe_inverse_float(x):
#     try:
#         return 1 / float(x)
#     except ValueError:
#         return np.nan
#     except ZeroDivisionError:
#         return np.nan
#
#
# # Parse the data.
# x, y = [], []
# with open('') as f:
#     reader = csv.reader(f)
#     header = next(reader)[3:]  # Skip the first three columns.
#     for row in reader:
#         dt = datetime.datetime.strptime(row[1], '%Y/%m/%d')
#         x.append(date_to_decimal_year(dt))
#         y.append([safe_inverse_float(c) for c in row[3:]])
#
# x = np.stack(x, axis=0)
# y = np.stack(y, axis=0)

def get_sim_data(a):

    #sampling time
    delta_t = 0.0165
    #select trial
    trial = a
    #get all data set
    # header = ["o4", "o5", "o6", "o7", "o8", "o9", "o10", "o11"]
    header = ["o7", "o8", "o9", "o10", "o11"]
    # all_datacsv = pd.read_csv('real_model_free.csv')
    all_datacsv = pd.read_csv('sim_regular_new2.csv')

    #get trial column
    trial_data = all_datacsv.values[:, 0]
    trial_column = np.where(trial_data == trial)

    #get frame column
    frame_data = all_datacsv.values[trial_column, 1] * delta_t
    # print ("trial column: ", trial_column)
    # print ("frame column: ", frame_data)

    #get o4~o11
    o_data = all_datacsv.values[trial_column, 41:49]
    # o_data = all_datacsv.values[trial_column, 44:49]
    # print ("o data: ", o_data)

    X_in = frame_data[0, 0:200]
    Y_out = o_data[0, 0:200, :]
    # print (type(X_in), X_in, type(Y_out), Y_out)
    return header, X_in, Y_out

def get_real_data(trail, min_len=1500):

    #sampling time
    delta_t = 0.0165
    #select trial
    trial = trail
    #get all data set
    # header = ["o4", "o5", "o6", "o7", "o8", "o9", "o10", "o11"]
    header = ["o7", "o8", "o9", "o10", "o11"]
    all_datacsv = pd.read_csv('real_model_free.csv')
    # all_datacsv = pd.read_csv('sim_regular_new2.csv')

    #get trial column
    trial_data = all_datacsv.values[:, 0]
    trial_column = np.where(trial_data == trial)
    print ("trial column: ", trial_column)

    #get o4~o11
    o_data = all_datacsv.values[trial_column, 40:48]

    zero_array = np.array([0,0,0,0])

    invalid_column = []
    # print (o_data[0,1,4:8])
    for i in range(o_data.shape[1]):
        if (o_data[0, i, 4:8] == zero_array).all():
            invalid_column.append(i)
    o_data= np.delete(o_data,invalid_column, axis=1)

    if o_data.shape[1] >= min_len:
        X_in = np.arange( 0, o_data.shape[1], 1 )[0:1400] * delta_t
        Y_out = o_data[0, 0:1400, :]
        # X_in = np.arange( 0, o_data.shape[1], 1 )[0:300] * delta_t
        # Y_out = o_data[0, 0:300, :]
        print (header, type(X_in), X_in.shape, type(Y_out), Y_out.shape)
        return header, X_in, Y_out

    else:
        print ("The length of data after filter is:", o_data.shape[1], "less than", min_len)
        print ("Quit!!!")
        os._exit(0)

# header, x, y = get_sim_data(6)
header, x, y = get_real_data(200, 1490)
# print ("o data: ", o_data[0])

# print(np.shape(x), np.shape(y))
# Reorder the data, putting the to be predicted outputs last.
#   Note: output 2 misses quite a lot of data.
to_predict = [header.index('o7')]#,
              # header.index('o7'),
              # header.index('o7'),
              # header.index('o7')]
              # header.index('o8'),
              # header.index('o9'),
              # header.index('o10'),
              # header.index('o11')]
print (range(len(header)), set(range(len(header))), set(to_predict))
others = sorted(set(range(len(header))) - set(to_predict))
order = others + to_predict
print(others, order, np.shape(order))
# Perform reordering of the data
y = y[:, order]
header = [header[i] for i in order]

# Remove regions from training data.
y_all = y.copy()
regions = [('o7', np.arange(301,1400), header.index('o7'))]#,
           # ('o7', np.arange(451,500), header.index('o7')),
           # ('o7', np.arange(451,500), header.index('o7')),
           # ('o7', np.arange(451,500), header.index('o7'))]

for _, inds, p in regions:
    y[inds, p] = np.nan

# Fit and predict GPAR.
model = GPARRegressor(scale=0.1,
                      linear=True, linear_scale=3.,
                      nonlinear=True, nonlinear_scale=0.2,
                      rq=False,
                      noise=1.,
                      impute=True, replace=True, normalise_y=True)
model.fit(x, y)
means, lowers, uppers = \
    model.predict(x, num_samples=200, credible_bounds=True, latent=False)

# Compute SMSEs.
smses = []
for _, inds, p in regions:
    # For the purpose of comparison, standardise using the mean of the
    # *training* data! This is *not* how the SMSE usually is defined.
    mse_mean = np.nanmean((y_all[inds, p] - np.nanmean(y[:, p])) ** 2)
    mse_gpar = np.nanmean((y_all[inds, p] - means[inds, p]) ** 2)
    smses.append(mse_gpar / mse_mean)
print('Average SMSE:', np.mean(smses))

# Plot the result.
plt.figure(figsize=(12, 3))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stixsans'

for i, (name, inds, p) in enumerate(regions):
    ax = plt.subplot(1, 2, i + 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.plot(x, means[:, p], c='blue')
    plt.fill_between(x, lowers[:, p], uppers[:, p],
                     facecolor='blue', alpha=.25)
    plt.scatter(x, y[:, p], c='green', marker='x', s=10)
    plt.scatter(x[inds], y_all[inds, p], c='orange', marker='x', s=10)
    plt.xlabel('Time (s)')
    plt.ylabel('State')
    plt.title(name)

plt.tight_layout()
plt.savefig('o7.png')
plt.show()