import os

get_wd = os.getcwd()
os.chdir(get_wd)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import wandb
import numpy as np
import pandas as pd
from utils import get_train_data
from PINTO_pde import PdeModel

np.random.seed(1234)
tf.random.set_seed(1234)

wandb.login(key="b52d765c1694df3d9a938427b8a0efec0d369688")

# Data PreProcessing
# getting data in required format from utils.py
data_dir = './dataset/train_64_40000.h5'  # change the directory to your local directory where Poisson data file is present.

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU:", gpus)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected, training will run on CPU")

# hyperparameters for data generation
train_indices = np.arange(80)
test_indices = [85, 90, 95]
val_indices = np.arange(80, 100)
domain_samples = 2000

# Get training data for Poisson equation
train_data = get_train_data(data_dir, domain_samples=domain_samples, 
                           indices=train_indices, val_indices=val_indices)

# Extract raw data before reshaping for context creation
ad_raw = train_data['ad']  # (nTrain, domain_samples)
fd_raw = train_data['fd']  # (nTrain, domain_samples)

# Prepare inputs and outputs for PINTO model
# For Poisson: inputs are (x, a, f) and output is p
ivals = {
    'xd': train_data['xd'].reshape(-1, 1),           # domain points for PDE residual
    'ad': train_data['ad'].reshape(-1, 1),           # coefficients at domain points
    'fd': train_data['fd'].reshape(-1, 1),           # forcing terms at domain points
    'x_bc': train_data['x_bc'].reshape(-1, 1),       # boundary points
    'p_bc': train_data['p_bc'].reshape(-1, 1),       # boundary values
    'a_bc': train_data['a_bc'].reshape(-1, 1),       # coefficients at boundary
    'x_init': train_data['x_init'].reshape(-1, 1),   # full grid points
    'p_init': train_data['p_init'].reshape(-1, 1),   # solutions on full grid
    'a_init': train_data['a_init'].reshape(-1, 1),   # coefficients on full grid
    'f_init': train_data['f_init'].reshape(-1, 1),   # forcing on full grid
    'x_val': train_data['x_val'].reshape(-1, 1),     # validation points
    'p_val': train_data['p_val'].reshape(-1, 1),     # validation solutions
    'a_val': train_data['a_val'].reshape(-1, 1),     # validation coefficients
    'f_val': train_data['f_val'].reshape(-1, 1)      # validation forcing
}

ovals = {
    'p_bc': train_data['p_bc'].reshape(-1, 1),       # boundary conditions
    'p_init': train_data['p_init'].reshape(-1, 1),   # supervised training data
    'p_val': train_data['p_val'].reshape(-1, 1)      # validation data
}

parameters = {'test_ind': test_indices}

# Building the PINTO model using functional API
initializer = tf.keras.initializers.GlorotUniform(seed=1234)


def get_model(model_name, layer_names, layer_units, activation='swish'):
    sq = keras.Sequential(name=model_name)
    for i in range(len(layer_units)):
        sq.add(layers.Dense(units=layer_units[i], kernel_initializer=initializer,
                            name=layer_names + str(i + 1)))
        sq.add(layers.Activation(activation=activation))
    return sq


# Lifting operator for query values (spatial coordinate)
input1 = layers.Input(shape=(1,), name='x_input')
rescale_input1 = layers.Rescaling(scale=2, offset=-1.)(input1)

# For Poisson: we only have spatial dimension, so we create a dummy dimension
# or use coefficient as a second dimension to maintain model structure
input2 = layers.Input(shape=(1,), name='a_input')  # coefficient a(x)
rescale_input2 = layers.Rescaling(scale=1., offset=0.)(input2)  # coefficients might not need rescaling

sp_trans = get_model(model_name='spatial_transformation',
                     layer_names='spatial_layer',
                     layer_units=[64, 64], activation='tanh')

sp = layers.Concatenate()([rescale_input1, rescale_input2])
sp = layers.Reshape(target_shape=(1, -1))(sp)
spq = sp_trans(sp)
residual = spq

# MLP for key values (coefficient and forcing term context)
# Using coefficient as context instead of initial coordinates
input3 = layers.Input(shape=(None, 1,), name='a_context')
rescale_input3 = layers.Rescaling(scale=1., offset=0)(input3)
input4 = layers.Input(shape=(None, 1,), name='f_context')  # forcing term
rescale_input4 = layers.Rescaling(scale=1., offset=0.)(input4)

pe = layers.Concatenate()([rescale_input3, rescale_input4])
pe = get_model(model_name='BPE',
               layer_names='bpe_layer',
               layer_units=[64, 64], activation='tanh')(pe)

# MLP for value values (using forcing term as values instead of initial conditions)
input5 = layers.Input(shape=(None, 1,), name='f_values')
ce = layers.Dense(units=64, kernel_initializer=initializer, activation='tanh',
                  name='bve_layer_1')(input5)
ce = layers.Dense(units=64, kernel_initializer=initializer, activation='tanh',
                  name='bve_layer_2')(ce)

# Cross Attention units (same architecture as original)
spk = layers.MultiHeadAttention(num_heads=2, key_dim=64)(query=spq, key=pe, value=ce)
spk = layers.Add()([residual, spk])
residual = spk
spk = layers.Dense(units=64, activation='tanh', kernel_initializer=initializer)(spk)
spk = layers.Dense(units=64, activation='tanh', kernel_initializer=initializer)(spk)
spk = layers.Add()([spk, residual])
residual = spk
spk = layers.MultiHeadAttention(num_heads=2, key_dim=64)(query=spk, key=pe, value=ce)
spk = layers.Add()([residual, spk])
spk = layers.Dense(units=64, activation='tanh', kernel_initializer=initializer)(spk)
ct = layers.Flatten()(spk)
residual = ct

# Projection operator for p function space (solution)
op = get_model(model_name='P', layer_units=[64, 64],
               layer_names='op', activation='tanh')(ct)
op = layers.Add()([residual, op])
op = layers.Dense(units=1, kernel_initializer=initializer, name='output_p')(op)

# building the PINTO model
# Inputs: [x, a(x), a_context, f_context, f_values]
# Output: p(x)
model = keras.Model([input1, input2, input3, input4, input5], [op])

# metrics to track the performance of the model during training
metrics = {"loss": keras.metrics.Mean(name='loss'),
           "bound_loss": keras.metrics.Mean(name='bound_loss'),
           "residual_loss": keras.metrics.Mean(name='residual_loss'),
           "val_loss": keras.metrics.Mean(name='val_loss'),
           "val_data_loss": keras.metrics.Mean(name='val_data_loss'),
           "val_res_loss": keras.metrics.Mean(name='val_res_loss'),
           }

# Training the model
initial_learning_rate = 1e-5

## Defining different Learning rate schedulers for different experiments
## Exponential Decay

# decay_steps = 10000
# decay_rate = 0.9
# staircase = True

# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=initial_learning_rate,
#     decay_steps=decay_steps,
#     decay_rate=decay_rate,
#     staircase=staircase)

# initiating the optimizer and loss function
optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
loss_fn = keras.losses.MeanSquaredError()

model.summary()
model_dict = {"nn_model": model}
batches = 10

# Prepare data for PINTO model based on the new input structure
# We need to create context data from the domain samples
n_samples = len(train_indices)
n_domain = domain_samples

# Create context data by sampling from the domain
# For simplicity, we'll use the first few domain points as context
context_size = min(60, n_domain)  # similar to seq_len in advection

# Sample context indices
context_indices = np.random.choice(n_domain, context_size, replace=False)

# Create context tensors for each sample
a_context_list = []
f_context_list = []
f_values_list = []

for i in range(n_samples):
    # Get context data for this sample - use the raw data with proper indexing
    a_ctx = ad_raw[i, context_indices].reshape(-1, 1)  # (context_size, 1)
    f_ctx = fd_raw[i, context_indices].reshape(-1, 1)  # (context_size, 1)
    
    a_context_list.append(a_ctx)
    f_context_list.append(f_ctx)
    f_values_list.append(f_ctx)  # Using same as values

# Stack and reshape for model input
a_context = np.stack(a_context_list, axis=0)  # (n_samples, context_size, 1)
f_context = np.stack(f_context_list, axis=0)
f_values = np.stack(f_values_list, axis=0)

# Repeat context for all domain points, boundary points, etc.
total_domain_points = len(ivals['xd'])
total_boundary_points = len(ivals['x_bc'])
total_init_points = len(ivals['x_init'])
total_val_points = len(ivals['x_val'])

# Calculate how many times to repeat each sample's context
domain_points_per_sample = domain_samples
boundary_points_per_sample = 2  # 2 boundary points per sample
init_points_per_sample = train_data['x_init'].shape[1]  # nx points per sample

# For domain points - repeat each sample's context for its domain points
a_context_domain_list = []
f_context_domain_list = []
f_values_domain_list = []

for i in range(n_samples):
    # Repeat context for all domain points of this sample
    sample_context_a = np.tile(a_context[i:i+1], (domain_points_per_sample, 1, 1))
    sample_context_f = np.tile(f_context[i:i+1], (domain_points_per_sample, 1, 1))
    sample_values_f = np.tile(f_values[i:i+1], (domain_points_per_sample, 1, 1))
    
    a_context_domain_list.append(sample_context_a)
    f_context_domain_list.append(sample_context_f)
    f_values_domain_list.append(sample_values_f)

a_context_domain = np.concatenate(a_context_domain_list, axis=0)  # (total_domain_points, context_size, 1)
f_context_domain = np.concatenate(f_context_domain_list, axis=0)
f_values_domain = np.concatenate(f_values_domain_list, axis=0)

# For boundary points
a_context_bc_list = []
f_context_bc_list = []
f_values_bc_list = []

for i in range(n_samples):
    # Repeat context for boundary points of this sample
    sample_context_a = np.tile(a_context[i:i+1], (boundary_points_per_sample, 1, 1))
    sample_context_f = np.tile(f_context[i:i+1], (boundary_points_per_sample, 1, 1))
    sample_values_f = np.tile(f_values[i:i+1], (boundary_points_per_sample, 1, 1))
    
    a_context_bc_list.append(sample_context_a)
    f_context_bc_list.append(sample_context_f)
    f_values_bc_list.append(sample_values_f)

a_context_bc = np.concatenate(a_context_bc_list, axis=0)
f_context_bc = np.concatenate(f_context_bc_list, axis=0)
f_values_bc = np.concatenate(f_values_bc_list, axis=0)

# For init points
a_context_init_list = []
f_context_init_list = []
f_values_init_list = []

for i in range(n_samples):
    # Repeat context for all init points of this sample
    sample_context_a = np.tile(a_context[i:i+1], (init_points_per_sample, 1, 1))
    sample_context_f = np.tile(f_context[i:i+1], (init_points_per_sample, 1, 1))
    sample_values_f = np.tile(f_values[i:i+1], (init_points_per_sample, 1, 1))
    
    a_context_init_list.append(sample_context_a)
    f_context_init_list.append(sample_context_f)
    f_values_init_list.append(sample_values_f)

a_context_init = np.concatenate(a_context_init_list, axis=0)
f_context_init = np.concatenate(f_context_init_list, axis=0)
f_values_init = np.concatenate(f_values_init_list, axis=0)

# For validation points
val_samples = len(val_indices)
val_points_per_sample = train_data['x_val'].shape[1]  # nx points per validation sample

a_context_val_list = []
f_context_val_list = []
f_values_val_list = []

# Create context for validation samples (use first val_samples contexts)
for i in range(min(val_samples, n_samples)):
    # Repeat context for all validation points of this sample
    sample_context_a = np.tile(a_context[i:i+1], (val_points_per_sample, 1, 1))
    sample_context_f = np.tile(f_context[i:i+1], (val_points_per_sample, 1, 1))
    sample_values_f = np.tile(f_values[i:i+1], (val_points_per_sample, 1, 1))
    
    a_context_val_list.append(sample_context_a)
    f_context_val_list.append(sample_context_f)
    f_values_val_list.append(sample_values_f)

a_context_val = np.concatenate(a_context_val_list, axis=0)
f_context_val = np.concatenate(f_context_val_list, axis=0)
f_values_val = np.concatenate(f_values_val_list, axis=0)

# Update ivals with context data
ivals.update({
    'a_context_domain': a_context_domain,
    'f_context_domain': f_context_domain,
    'f_values_domain': f_values_domain,
    'a_context_bc': a_context_bc,
    'f_context_bc': f_context_bc,
    'f_values_bc': f_values_bc,
    'a_context_init': a_context_init,
    'f_context_init': f_context_init,
    'f_values_init': f_values_init,
    'a_context_val': a_context_val,
    'f_context_val': f_context_val,
    'f_values_val': f_values_val,
})

# initiating the PdeModel class
cm = PdeModel(inputs=ivals, outputs=ovals, get_models=model_dict, loss_fn=loss_fn,
              optimizer=optimizer, metrics=metrics,
              parameters=parameters, batches=batches)

epochs = 10000
vf = 100  # verbose frequency
pf = 1000  # plot frequency
wb = True  # wandb logging

configuration = {
    '#_total_boundary_points': len(ivals['x_bc']),
    '#_total_domain_points': len(ivals['xd']),
    "optimizer": "Adam",
    'initial_learning_rate': initial_learning_rate,
    # 'lr_Schedule': 'Exponential Decay',
    # 'decay_steps': decay_steps,
    # 'decay_rate': decay_rate,
    # 'staircase': staircase,
    "batches": batches,
    "Epochs": epochs,
    "Activation": 'swish',
    "model_name": 'Poisson_PINTO_model',
    "trainable_parameters": np.sum([np.prod(lay.shape) for lay in model.trainable_weights]),
    "non_trainable_parameters": np.sum([np.prod(lay.shape) for lay in model.non_trainable_weights]),
    'test_indices': test_indices,
    "context_size": context_size,
    "domain_samples": domain_samples}

print(configuration)

if wb:
    wandb.init(project='Poisson_PINTO', config=configuration)

log_dir = 'output/Poisson_PINTO/'
try:
    os.makedirs(log_dir)
except FileExistsError:
    pass

history = cm.run(epochs=epochs, ddir=data_dir, log_dir=log_dir,
                 wb=wb, verbose_freq=vf, plot_freq=pf)

context_data = pd.DataFrame({'context_indices': context_indices.flatten()})
context_data.to_csv(path_or_buf=log_dir + 'context.csv')

if wb:
    wandb.finish()

# Evaluation
cm.nn_model.save(log_dir + 'Poisson_model.keras')
