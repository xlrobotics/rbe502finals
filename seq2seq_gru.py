
import tensorflow as tf  # Version 1.0 or 0.12
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from RNN_data_read import RNNdata as rd
from RNN_data_read import get_training_data
from copy import deepcopy as dcp

# This is for the notebook to generate inline matplotlib 
# charts rather than to open a new window every time: 
# get_ipython().magic('matplotlib inline')


# ## Neural network's hyperparameters

# Internal neural network parameters
seq_length = 9  # sample_x.shape[0]  # Time series will have the same past and future (to be predicted) length.
seq_length_y = 3

batch_size = 64  # Low value used for live demo purposes - 100 and 1000 would be possible too, crank that up!

input_dim = 8
output_dim = 4  # Output dimension (e.g.: multiple signals at once, tied in time)

hidden_dim = 1024  #800  # 1024  # Count of hidden neurons in the recurrent units. XL: also contributes to avoid overfitting
layers_stacked_count = 3  # 3 # XL: the depth of layers might relate to the polynomial order of the data,
# Optmizer: 
learning_rate = 0.0003  # Small lr helps not to diverge during training.
nb_iters = 24000  # 30000  # 25000  # How many times we perform a training step (therefore how many times we show a batch).
lr_decay = 0.95  # 0.92  # default: 0.9 . Simulated annealing.
momentum = 0.5  # default: 0.0 . Momentum technique in weights update
lambda_l2_reg = 0.001  # 0.003  #  L2 regularization of weights - avoids overfitting


# ## Definition of the seq2seq neuronal architecture

# Backward compatibility for TensorFlow's version 0.12: 
try:
    tf.nn.seq2seq = tf.contrib.legacy_seq2seq
    tf.nn.rnn_cell = tf.contrib.rnn
    tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
    print("TensorFlow's version : 1.0 (or more)")
except: 
    print("TensorFlow's version : 0.12")


tf.reset_default_graph()
# sess.close()
sess = tf.InteractiveSession()

with tf.variable_scope('Seq2seq'):

    # Encoder: inputs
    enc_inp = [
        tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
        for t in range(seq_length)
    ]

    # Decoder: expected outputs
    expected_sparse_output = [
        tf.placeholder(tf.float32, shape=(None, output_dim), name="expected_sparse_output_".format(t))
        for t in range(seq_length_y)
    ]
    
    # Give a "GO" token to the decoder. 
    # Note: we might want to fill the encoder with zeros or its own feedback rather than with "+ enc_inp[:-1]"
    dec_inp = [
        tf.zeros_like(expected_sparse_output[0], dtype=np.float32, name="GO")
    ] + expected_sparse_output[:-1]  # inc_inp

    # dec_inp = [
    #     tf.zeros_like(expected_sparse_output[0], dtype=np.float32, name="GO")
    # ] + expected_sparse_output[0]

    # Create a `layers_stacked_count` of stacked RNNs (GRU cells here). 
    cells = []
    for i in range(layers_stacked_count):
        with tf.variable_scope('RNN_{}'.format(i)):
            # cells.append(tf.nn.rnn_cell.BasicRNNCell(hidden_dim))
            # basic_cell = tf.nn.rnn_cell.DropoutWrapper(
            #             tf.nn.rnn_cell.GRUCell(hidden_dim),
            #             output_keep_prob=tf.placeholder(tf.float32))
            # cells.append(basic_cell)
            # cells.append(tf.nn.rnn_cell.GRUCell(hidden_dim))
            cells.append(tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(hidden_dim)
                                                       , output_keep_prob=0.90))

            # cells.append(tf.nn.rnn_cell.LSTMCell(hidden_dim))
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    
    # Here, the encoder and the decoder uses the same cell, HOWEVER,
    # the weights aren't shared among the encoder and decoder, we have two
    # sets of weights created under the hood according to that function's def. 
    dec_outputs, dec_memory = tf.nn.seq2seq.basic_rnn_seq2seq(
        enc_inp, 
        dec_inp, 
        cell
    )
    
    # For reshaping the output dimensions of the seq2seq RNN: # dense net to the output
    w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
    b_out = tf.Variable(tf.random_normal([output_dim]))
    
    # Final outputs: with linear rescaling for enabling possibly large and unrestricted output values.
    output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
    
    reshaped_outputs = [output_scale_factor*(tf.matmul(i, w_out) + b_out) for i in dec_outputs]


# Training loss and optimizer

with tf.variable_scope('Loss'):
    # L2 loss
    output_loss = 0
    for _y, _Y in zip(reshaped_outputs, expected_sparse_output):
        output_loss += tf.reduce_mean(tf.nn.l2_loss(_y - _Y))
        
    # L2 regularization (to avoid overfitting and to have a  better generalization capacity)
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
            reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
            
    loss = output_loss + lambda_l2_reg * reg_loss

with tf.variable_scope('Optimizer'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=lr_decay, momentum=momentum)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)


total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    print(shape)
    print(len(shape))
    variable_parameters = 1
    for dim in shape:
        print(dim)
        variable_parameters *= dim.value
    print(variable_parameters)
    total_parameters += variable_parameters
print(total_parameters)


# ## Importing data
np.random.seed(5) #10
dataX, dataY = rd(length=seq_length, length_ = seq_length_y)
total_index = np.arange(len(dataX))
np.random.shuffle(total_index)

# ## Training of the neural net
def train_batch(k, batch_size):
    """
    Training step that optimizes the weights
    provided some batch_size X and Y examples from the dataset.
    """
    X, Y = get_training_data(dataX, dataY, total_index[k:k+batch_size])
    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Y[t] for t in range(len(expected_sparse_output))})
    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t, k+batch_size


def test_batch(k, batch_size):
    """
    Test step, does NOT optimizes. Weights are frozen by not
    doing sess.run on the train_op.
    """

    X, Y = get_training_data(dataX, dataY, total_index[k:k+batch_size])
    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Y[t] for t in range(len(expected_sparse_output))})
    loss_t = sess.run([loss], feed_dict)
    return loss_t[0], k+batch_size

# Training
train_losses = []
test_losses = []

# Restoration of previously trained model
# meta_path = 'logs/gru_model/gru_model.ckpt-50000.meta'
# model_path = 'logs/gru_model/gru_model.ckpt-50000'
# saver = tf.train.import_meta_graph(meta_path)
# config = tf.ConfigProto()
# restore = True
options = tf.global_variables_initializer()

# Saver Initialization
saver = tf.train.Saver()

sess.run(options)

current_k = 0
for t in range(nb_iters+1):
    train_loss, current_k = train_batch(current_k, batch_size)
    train_losses.append(train_loss/batch_size)

    if t % int(nb_iters/5) == 0:
        save_path = saver.save(sess, save_path='logs/gru_model/gru_model.ckpt', global_step=t)
        print("Model saved in path: %s" % save_path)

    if t % 10 == 0:
        # Tester
        test_loss, current_k = test_batch(current_k, batch_size)
        test_losses.append(test_loss/batch_size)
        print("Step {}/{}, train loss: {}, \tTEST loss: {}, data_used {}/{}".format(t, nb_iters, train_loss, test_loss, current_k, len(total_index)))

print("Fin. train loss: {}, \tTEST loss: {}".format(train_loss, test_loss))

# Plot loss over time:
mpl.style.use("seaborn")
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(
    np.array(range(0, len(test_losses)))/float(len(test_losses)-1)*(len(train_losses)-1), 
    np.log(test_losses),
    label="Test loss",
    color="g",
)
plt.grid(True)
plt.title("Testing loss over time (on a logarithmic scale)")
plt.xlabel('Iteration')
plt.ylabel('log(Loss)')
plt.legend(loc='best')
plt.subplot(2, 1, 2)
plt.plot(
    np.log(train_losses),
    label="Train loss",
    color="r",
    #linestyle='dashed',
)
plt.grid(True)
plt.title("Training loss over time (on a logarithmic scale)")
plt.xlabel('Iteration')
plt.ylabel('log(Loss)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# In[9]:


# plot testing result examples
# nb_predictions = 10
# print("Let's visualize {} predictions with our signals:".format(nb_predictions))
#
# start_x = 130  # 130000 #4*220000  #3750000 #
# X, Y = get_training_data(dataX, dataY, total_index[start_x:start_x+nb_predictions])
# # X, Y = generate_x_y_data(isTrain=False, batch_size=nb_predictions)
# Y_zeros = Y*0.0
#
# feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
# feed_dict.update({expected_sparse_output[t]: Y_zeros[t] for t in range(len(expected_sparse_output))})
# outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])
#
# for j in range(nb_predictions):
#     plt.figure(figsize=(9, 12))
#
#     for k in range(0, 4):
#         plt.subplot(4, 1, k+1)
#         past = X[:, j, k]
#         expected = np.concatenate((X[-1:, j, k], Y[:, j, k]))
#         pred = np.concatenate((X[-1:, j, k], outputs[:, j, k]))  # outputs[:, j, k]
#
#         label1 = "Seen (past) values" if k==0 else "_nolegend_"
#         label2 = "True future values" if k==0 else "_nolegend_"
#         label3 = "Predictions" if k==0 else "_nolegend_"
#
#         plt.plot(range(len(past)-1, len(expected)+len(past)-1), expected, "*--g", label=label2, markersize=9)
#         plt.plot(range(len(past)-1, len(pred)+len(past)-1), pred, "*--y", label=label3, markersize=9)
#         plt.plot(range(len(past)), past, "*--b", label=label1, markersize=9)
#
#         plt.legend(loc='best')
#         plt.title("Predictions v.s. true values on "+str(k+1)+"th link curvature.")
#         plt.ylim(bottom=-1)
#         plt.ylim(top=1)
#
#     plt.tight_layout()
#     plt.show()


# plot long prediction frames
# np.random.seed(5) #10
# dataX, dataY = rd(length=seq_length, length_ = seq_length_y)
# total_index = np.arange(len(dataX))
# np.random.shuffle(total_index)

original_index = np.arange(len(dataX))
start_x = 200

steps = 500
X, Y = get_training_data(dataX, dataY, original_index[start_x:start_x+steps])

Y_pred = dcp(Y[:, 0, :])
Y_true = dcp(Y[:, 0, :])
# X_current = X[0]
# Y_current = Y[0]
X_current = X[:, 0:2, :]
Y_current = Y[:, 0:2, :]

Y_zeros = Y_current*0.0

for s in range(1, steps-2):
# s = 1
# while s < steps-seq_length_y:
    feed_dict = {enc_inp[t]: X_current[t] for t in range(seq_length)}
    feed_dict.update({expected_sparse_output[t]: Y_zeros[t] for t in range(len(expected_sparse_output))})
    outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])

    # X_current = np.concatenate((X_current[1:,:,0:4],outputs[]))
    temp = np.concatenate((outputs[-1:, :, :], X[-1:, s:s + 2, 4:8]), axis=2)
    X_current = np.concatenate((X_current[1:, :, :], temp))
    Y_current = Y[-1:, s:s+2, :]
    Y_pred = np.concatenate((Y_pred, outputs[-1:, 0, :]))
    Y_true = np.concatenate((Y_true, Y_current[-1:, 0, :]))

    # s += seq_length_y

plt.figure(figsize=(24, 12))
for k in range(0, 4):
    plt.subplot(4, 1, k + 1)
    past = X[:, 0, :]
    expected = np.concatenate((past[:, k], Y_true[:, k]))
    pred = np.concatenate((past[:, k], Y_pred[:, k]))  # outputs[:, j, k]

    # label1 = "Seen (past) values" if k == 0 else "_nolegend_"
    label1 = "True future values" if k == 0 else "_nolegend_"
    label2 = "Predictions" if k == 0 else "_nolegend_"

    plt.plot(range(0, len(expected)), expected, "^--g", label=label1, markersize=5)
    plt.plot(range(0, len(pred)), pred, "v--y", label=label2, markersize=5)
    # plt.plot(range(len(past)), past, "*--b", label=label1, markersize=9)

    plt.legend(loc='best')
    plt.title("Predictions v.s. true values on " + str(k + 1) + "th link curvature.")
    plt.ylim(bottom=-1)
    plt.ylim(top=1)

plt.tight_layout()
plt.show()