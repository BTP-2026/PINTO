import os
get_wd = os.getcwd()
os.chdir(get_wd)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import wandb
import numpy as np
import pandas as pd
from utils import get_train_data
from be1d_Pde import PdeModel

np.random.seed(1234)
tf.random.set_seed(1234)

# Data PreProcessing
data_dir = '../../burgers_1d_t10.mat'

train_indices = np.arange(80)
test_indices = [10, 20, 85, 99]
val_indices = np.arange(80, 100)
domain_samples = 2000
sensor_samples = 40

(xd, td, xb, tb, ub, x_init, t_init, u_init,
 xbc_in, tbc_in, ubc_in, xbc_b, tbc_b, ubc_b, xbc_init, tbc_init, ubc_init,
 idx_si, xval, tval, uval, xbc_val, tbc_val, ubc_val) = get_train_data(
    data_dir, sensor_samples=sensor_samples, domain_samples=domain_samples, indices=train_indices,
    val_indices=val_indices)

ivals = {'xin': xd, 'tin': td, 'xb': xb, 'tb': tb, 'xbc_in': xbc_in, 'tbc_in': tbc_in, 'ubc_in': ubc_in,
         'xbc_b': xbc_b, 'tbc_b': tbc_b, 'ubc_b': ubc_b, 'x_init': x_init, 't_init': t_init,
         'xbc_init': xbc_init, 'tbc_init': tbc_init, 'ubc_init': ubc_init,
         'xval': xval, 'tval': tval, 'xbc_val': xbc_val, 'tbc_val': tbc_val, 'ubc_val': ubc_val}
ovals = {'ub': ub, 'u_init': u_init, 'uval': uval}
parameters = {'nue': 0.01, 'test_ind': test_indices}

initializer = tf.keras.initializers.GlorotUniform(seed=1234)


def get_model(model_name, layer_names, layer_units, activation='swish'):
    sq = keras.Sequential(name=model_name)
    for i in range(len(layer_units)):
        sq.add(layers.Dense(units=layer_units[i], kernel_initializer=initializer,
                            name=layer_names + str(i + 1)))
        sq.add(layers.Activation(activation=activation))
    return sq


# with strategy.scope():
input1 = layers.Input(shape=(1,), name='x_input')
rescale_input1 = layers.Rescaling(scale=2.0, offset=-1.)(input1)
input2 = layers.Input(shape=(1,), name='t_input')
rescale_input2 = layers.Rescaling(scale=2.0, offset=-1.)(input2)

# transforming query values
sp_trans = get_model(model_name='spatial_transformation',
                     layer_names='spatial_layer',
                     layer_units=[64, 64], activation='tanh')

sp = layers.Concatenate()([rescale_input1, rescale_input2])
sp = layers.Reshape(target_shape=(1, -1))(sp)
spq = sp_trans(sp)
residual = spq

# key transformation for boundary
input3 = layers.Input(shape=(None, 1,), name='Xbc_layer')
rescale_input3 = layers.Rescaling(scale=2.0, offset=-1.)(input3)
input4 = layers.Input(shape=(None, 1,), name='tbc_layer')
rescale_input4 = layers.Rescaling(scale=2.0, offset=-1.)(input4)

# position encoding
pe = layers.Concatenate()([rescale_input3, rescale_input4])
pe = get_model(model_name='BPE',
               layer_names='bpe_layer',
               layer_units=[64, 64], activation='tanh')(pe)

# value transformation for geometry
input5 = layers.Input(shape=(None, 1,), name='ubc_layer')
ce = get_model(model_name='BVE',
               layer_names='bve_layer',
               layer_units=[64, 64], activation='tanh')(input5)

ct = layers.MultiHeadAttention(num_heads=2, key_dim=64)(query=spq, key=pe, value=ce)
ct = layers.Add()([residual, ct])
residual = ct
ct = layers.Dense(units=64, activation='tanh', kernel_initializer=initializer)(ct)
ct = layers.Dense(units=64, activation='tanh', kernel_initializer=initializer)(ct)
ct = layers.Add()([residual, ct])
residual = ct
ct = layers.MultiHeadAttention(num_heads=2, key_dim=64)(query=ct, key=pe, value=ce)
ct = layers.Add()([residual, ct])
residual = ct
ct = layers.Dense(units=64, activation='tanh', kernel_initializer=initializer)(ct)
ct = layers.Dense(units=64, activation='tanh', kernel_initializer=initializer)(ct)
ct = layers.Add()([residual, ct])
residual = ct
ct = layers.MultiHeadAttention(num_heads=2, key_dim=64)(query=ct, key=pe, value=ce)
ct = layers.Add()([residual, ct])
ct = layers.Dense(units=64, activation='tanh', kernel_initializer=initializer)(ct)
ct = layers.Flatten()(ct)
residual = ct

ou = get_model(model_name='U', layer_units=[64, 64],
               layer_names='ou', activation='tanh')(ct)
ou = layers.Dense(units=1, kernel_initializer=initializer, name='output_u')(ou)

model = keras.Model([input1, input2, input3, input4, input5],
                    [ou])

metrics = {"loss": keras.metrics.Mean(name='loss'),
           "bound_loss": keras.metrics.Mean(name='bound_loss'),
           "init_loss": keras.metrics.Mean(name='init_loss'),
           "residual_loss": keras.metrics.Mean(name='residual_loss'),
           "val_loss": keras.metrics.Mean(name="val_loss"),
           "val_data_loss": keras.metrics.Mean(name="val_data_loss"),
           "val_res_loss": keras.metrics.Mean(name="val_res_loss")
           }

# Training the model
initial_learning_rate = 1e-3

decay_steps = 10000
decay_rate = 0.9
staircase = True
#
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=staircase)

# lr_schedule = MyRLSchedule(
#     initial_learning_rate=initial_learning_rate,
#     decay_steps=decay_steps,
#     higher_decay_rate=decay_rate[0],
#     lower_decay_rate=decay_rate[1],
#     staircase=staircase, step_lim=step_lim)

optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
loss_fn = keras.losses.MeanSquaredError()

model.summary()
model.compile(optimizer=optimizer, loss=loss_fn)
model_dict = {"nn_model": model}
batches = 10

cm = PdeModel(inputs=ivals, outputs=ovals, get_models=model_dict, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics,
              parameters=parameters, batches=batches)

epochs = 10
vf = 1
pf = 10
valf = None
wb = False

configuration = {
    '#_total_initial_and_boundary_points': len(xb),
    '#_total_domain_points': len(xd),
    "optimizer": "Adam",
    'initial_learning_rate': initial_learning_rate,
    'lr_Schedule': 'Exponential Decay',
    'decay_steps': decay_steps,
    'decay_rate': decay_rate,
    'staircase': staircase,
    "batches": batches,
    "Epochs": epochs,
    "kernel_initializer": 'GlorotUniform',
    "Activation": 'tanh',
    "model_name": 'Burgers_model.keras',
    'test_indices': test_indices,
    'sequence_length': sensor_samples,
    'nue': parameters['nue']}

print(configuration)

if wb:
    wandb.init(project='Finalising_results', config=configuration)

log_dir = 'output/Burgers_PINTO/'

history = cm.run(epochs=epochs, idx_si=idx_si, ddir=data_dir,
                 log_dir=log_dir, wb=wb, verbose_freq=vf, plot_freq=pf, validation_freq=valf)

sdata = pd.DataFrame({'sensor_indices': idx_si.flatten()})
sdata.to_csv(path_or_buf=log_dir + 'sensor.csv')

if wb:
    wandb.finish()

# Evaluation
cm.nn_model.save(log_dir + 'Burgers_model.keras')
