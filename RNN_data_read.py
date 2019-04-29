import csv
import numpy as np
import pandas as pd
from scipy import signal as sg


def lpf(data):

    # b, a = sg.butter(4, 0.03, analog=False)
    b, a = sg.butter(5, 0.35, analog=False)

    # Show that frequency response is the same
    # impulse = np.zeros(1000)
    # impulse[500] = 1

    # Applies filter forward and backward in time
    # imp_ff = sg.filtfilt(b, a, impulse)
    sig_ff = []
    for i in range(len(data[0])-4):
        sig_ff.append(sg.filtfilt(b, a, data[:, i]))
    for i in range(len(data[0])-4, len(data[0])):
        sig_ff.append(data[:, i])

    result = np.array(sig_ff).transpose((1, 0))
    return result


def RNNdata(length=5, length_=3, len_episode=1500, filter=False):  # 2700
    # data csv read
    datacsv = pd.read_csv('real_model_free.csv')
    # datacsv = pd.read_csv('sim_regular_1.csv')
    # datacsv = pd.read_csv('sim_regular_new2.csv')
    # datacsv = pd.read_csv('sim_regular_new.csv')
    datacsv1 = datacsv.values
    r, c = np.shape(datacsv1)
    datacsv2 = datacsv1[0, :]

    Xin = []
    Yout = []

    last_i = 0

    for i in range(r):
        # filter: non-zero

        if not datacsv1[i, 0] == datacsv1[i-1, 0]:

            if i-last_i < len_episode:
                last_i = i
                datacsv2 = datacsv1[i, :]
                continue
            # take o0-o11
            print('Loaded episode:', datacsv1[i-1, 0], "( Length:", i-last_i, " )")

            last_i = i

            if filter:
                dataRNN = lpf(datacsv2[:, 40:48])  # change to 41:49 if using the simulation data
            else:
                dataRNN = datacsv2[:, 40:48]  # change to 41:49 if using the simulation data

            r1, c1 = np.shape(dataRNN)

            # RNN Xin, Yout
            for i in range(r1):
                if i + 2 * length - 1 <= r1 - 1:
                    Xin.append(dataRNN[i:i + length, :])
                    Yout.append(dataRNN[i + length:i + length + length_, 0:4])

            datacsv2 = datacsv1[i, :]

        else:
            datacsv2 = np.vstack((datacsv2, datacsv1[i, :]))

    Xin.extend(Xin)
    Yout.extend(Yout)
    Xin.extend(Xin)
    Yout.extend(Yout)
    Xin.extend(Xin)
    Yout.extend(Yout)
    Xin = np.array(Xin)
    Yout = np.array(Yout)

    return Xin, Yout


def get_training_data(Xin, Yout, index):
    batch_x = []
    batch_y = []

    for element in index:
        batch_x.append(Xin[element])
        batch_y.append(Yout[element])

    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))

    return batch_x, batch_y


if __name__ == '__main__':

    batch = input("batchinput:")
    dataRNNin, dataRNNout = RNNdata(int(batch))

    total_index = np.arange(len(dataRNNin))
    np.random.shuffle(total_index)

    bx, by = get_training_data(Xin=dataRNNin, Yout=dataRNNout, index=total_index[0:100])

    print(type(dataRNNin))
    print(np.shape(dataRNNin), np.shape(dataRNNout))