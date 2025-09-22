import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import time
import wandb

from tensorflow.python.ops.numpy_ops import np_config
from utils import read_h5_file

np_config.enable_numpy_behavior()
tf.random.set_seed(1234)


class PdeModel:
    def __init__(self, inputs, outputs, get_models, loss_fn, optimizer, metrics, parameters,
                 batches=1, val_batches=50):

        self.inputs = inputs
        self.outputs = outputs
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batches = batches
        self.parameters = parameters

        # Create efficient data pipelines for Poisson equation
        # Domain points for PDE residual
        self.inner_data = self.create_data_pipeline(
            inputs['xd'], inputs['ad'], inputs['fd'],
            inputs['a_context_domain'], inputs['f_context_domain'], inputs['f_values_domain'],
            batch=batches).cache()
        
        # Boundary conditions
        self.bound_data = self.create_data_pipeline(
            inputs['x_bc'], inputs['p_bc'], inputs['a_bc'],
            inputs['a_context_bc'], inputs['f_context_bc'], inputs['f_values_bc'],
            batch=batches).cache()
        
        # Full grid data for validation/supervised learning
        self.init_data = self.create_data_pipeline(
            inputs['x_init'], inputs['p_init'], inputs['a_init'], inputs['f_init'],
            inputs['a_context_init'], inputs['f_context_init'], inputs['f_values_init'],
            batch=batches).cache()
        
        # Validation data
        self.val_data = self.create_data_pipeline(
            inputs['x_val'], inputs['p_val'], inputs['a_val'], inputs['f_val'],
            inputs['a_context_val'], inputs['f_context_val'], inputs['f_values_val'],
            batch=val_batches).cache()

        self.nn_model = get_models['nn_model']

        # Metrics tracking
        self.loss_tracker = metrics['loss']
        self.bound_loss_tracker = metrics['bound_loss']
        self.residual_loss_tracker = metrics['residual_loss']
        self.val_loss_tracker = metrics['val_loss']
        self.val_data_loss_tracker = metrics['val_data_loss']
        self.val_residual_loss_tracker = metrics['val_res_loss']

    @staticmethod
    def create_data_pipeline(*args, batch):
        dataset = tf.data.Dataset.from_tensor_slices(args)
        dataset = dataset.shuffle(buffer_size=len(args[0]))
        dataset = dataset.batch(np.ceil(len(args[0]) / batch))
        return dataset

    @tf.function
    def Pde_residual(self, input_data, training=True):
        """
        Compute Poisson equation residual: -d/dx(a(x)*dp/dx) - f(x) = 0
        """
        x, a, f, a_context, f_context, f_values = input_data

        with tf.GradientTape() as tape:
            tape.watch(x)
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                # Forward pass to get solution p(x)
                p = self.nn_model([x, a, a_context, f_context, f_values], training=training)
            
            # First derivative dp/dx
            dp_dx = tape2.gradient(p, x)
            
            # Product a(x) * dp/dx
            a_dp_dx = a * dp_dx

        # Second derivative term: d/dx(a(x)*dp/dx)
        d_a_dp_dx = tape.gradient(a_dp_dx, x)

        # Handle None gradients
        if d_a_dp_dx is None:
            # If gradient computation fails, return zero residual
            # This prevents training from crashing but may affect learning
            print("Warning: Gradient computation failed, using zero residual")
            residual_loss = tf.zeros_like(f)
            return residual_loss

        # Poisson equation: -d/dx(a(x)*dp/dx) = f(x)
        # Residual: -d/dx(a(x)*dp/dx) - f(x) = 0
        residual = -d_a_dp_dx - f
        residual_loss = tf.square(residual)
        return residual_loss

    @tf.function
    def train_step(self, bound_data, inner_data, init_data):
        """
        Training step for Poisson equation
        """
        x_bc, p_bc, a_bc, a_context_bc, f_context_bc, f_values_bc = bound_data
        x_init, p_init, a_init, f_init, a_context_init, f_context_init, f_values_init = init_data

        with tf.GradientTape(persistent=True) as tape:
            # Boundary condition predictions
            p_bc_pred = self.nn_model([x_bc, a_bc, a_context_bc, f_context_bc, f_values_bc], training=True)
            
            # Supervised learning on full grid
            p_init_pred = self.nn_model([x_init, a_init, a_context_init, f_context_init, f_values_init], training=True)

            # PDE residual on domain points
            residual_loss = tf.reduce_mean(self.Pde_residual(inner_data, training=True))

            # Boundary loss
            bound_loss = self.loss_fn(p_bc, p_bc_pred)
            
            # Supervised loss (optional, can be weighted)
            supervised_loss = self.loss_fn(p_init, p_init_pred)

            # Total loss: residual + boundary (+ optional supervised)
            loss = residual_loss + bound_loss

        grads = tape.gradient(loss, self.nn_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.nn_model.trainable_weights))

        del tape

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.bound_loss_tracker.update_state(bound_loss)
        self.residual_loss_tracker.update_state(residual_loss)

        return {
            "loss": self.loss_tracker.result(),
            "bound_loss": self.bound_loss_tracker.result(),
            "residual_loss": self.residual_loss_tracker.result()
        }

    @tf.function
    def test_step(self, val_data):
        """
        Validation step
        """
        x_val, p_val, a_val, f_val, a_context_val, f_context_val, f_values_val = val_data
        
        # Predictions
        p_pred = self.nn_model([x_val, a_val, a_context_val, f_context_val, f_values_val], training=False)
        
        # Data loss
        val_data_loss = self.loss_fn(p_val, p_pred)
        
        # Residual loss
        val_res_loss = tf.reduce_mean(
            self.Pde_residual([x_val, a_val, f_val, a_context_val, f_context_val, f_values_val], training=False)
        )
        
        val_loss = val_data_loss + val_res_loss

        self.val_loss_tracker.update_state(val_loss)
        self.val_data_loss_tracker.update_state(val_data_loss)
        self.val_residual_loss_tracker.update_state(val_res_loss)
        
        return {
            'val_loss': self.val_loss_tracker.result(),
            'val_data_loss': self.val_data_loss_tracker.result(),
            'val_res_loss': self.val_residual_loss_tracker.result()
        }

    def reset_metrics(self):
        """Reset all metric states"""
        self.loss_tracker.reset_state()
        self.residual_loss_tracker.reset_state()
        self.bound_loss_tracker.reset_state()
        self.val_loss_tracker.reset_state()
        self.val_data_loss_tracker.reset_state()
        self.val_residual_loss_tracker.reset_state()

    def get_model_graph(self, log_dir, wb=False):
        """Model graph visualization (placeholder)"""
        pass

    def run(self, epochs, ddir, log_dir, wb=False, verbose_freq=1000, plot_freq=10000,
            validation_freq=1000):
        """
        Main training loop
        """
        history = {"loss": [], "residual_loss": [], "bound_loss": []}
        val_history = {"val_loss": [], "val_data_loss": [], "val_res_loss": []}

        self.get_model_graph(log_dir=log_dir, wb=wb)
        
        # Load full dataset for plotting
        p_all, a_all, f_all, bc_all, xdisc, L = read_h5_file(ddir)

        for epoch in range(epochs):
            start_time = time.time()
            self.reset_metrics()

            # Training step
            for j, (bound_data, inner_data, init_data) in enumerate(zip(
                    self.bound_data, self.inner_data, self.init_data)):
                logs = self.train_step(bound_data, inner_data, init_data)

            if wb:
                wandb.log(logs, step=epoch + 1)

            # Validation step
            if (epoch + 1) % validation_freq == 0:
                for j, val_data in enumerate(self.val_data):
                    val_logs = self.test_step(val_data)
                if wb:
                    wandb.log(val_logs, step=epoch + 1)

            # Record metrics
            elapsed_time = time.time() - start_time
            for key, value in logs.items():
                history[key].append(value.numpy())
            
            if (epoch + 1) % validation_freq == 0:
                for key, value in val_logs.items():
                    val_history[key].append(value.numpy())

            # Print progress
            print(f'Epoch:{epoch + 1}/{epochs}')
            for key, value in logs.items():
                print(f"{key}: {value:.4f} ", end="")
            if (epoch + 1) % validation_freq == 0:
                for key, value in val_logs.items():
                    print(f"{key}: {value:.4f} ", end="")
            print(f"Time: {elapsed_time / 60:.4f}min")

            # Generate plots
            if (epoch + 1) % plot_freq == 0:
                test_indices = self.parameters.get('test_ind', [0, 1, 2])  # Default test indices
                for i in test_indices:
                    if i < len(p_all):
                        self.get_plots(epoch + 1, xdisc, p_all[i], a_all[i], f_all[i], 
                                     log_dir=log_dir, ind=i, wb=wb)

        # Save training history
        odata = pd.DataFrame(history)
        val_odata = pd.DataFrame(val_history)
        odata.to_csv(path_or_buf=log_dir + 'history.csv')
        val_odata.to_csv(path_or_buf=log_dir + 'val_history.csv')

        # Plot loss curve
        plt.figure()
        plt.plot(range(1, len(odata) + 1), np.log(odata['loss']))
        plt.xlabel('Epochs')
        plt.ylabel('Log_Loss')
        plt.title('Log Loss Plot')
        plt.savefig(log_dir + '_log_loss_plt.png', dpi=300)
        if wb:
            wandb.log({"loss_plot": wandb.Image(log_dir + '_log_loss_plt.png')}, step=epochs)
        
        return history

    def predictions(self, inputs):
        """Generate predictions"""
        p_pred = self.nn_model.predict(inputs, batch_size=32, verbose=False)
        return p_pred

    def get_plots(self, step, xdisc, p_true, a_true, f_true, log_dir, ind, wb=False):
        """
        Generate plots comparing true vs predicted solutions
        """
        # Prepare inputs for prediction
        x_plot = xdisc.reshape(-1, 1)
        a_plot = a_true.reshape(-1, 1)
        f_plot = f_true.reshape(-1, 1)
        
        # Create dummy context data for plotting (use the true data as context)
        context_size = min(60, len(xdisc))
        context_indices = np.linspace(0, len(xdisc)-1, context_size, dtype=int)
        a_context_plot = a_true[context_indices].reshape(1, -1, 1)
        f_context_plot = f_true[context_indices].reshape(1, -1, 1)
        f_values_plot = f_true[context_indices].reshape(1, -1, 1)
        
        # Repeat context for all points
        a_context_repeated = np.tile(a_context_plot, (len(x_plot), 1, 1))
        f_context_repeated = np.tile(f_context_plot, (len(x_plot), 1, 1))
        f_values_repeated = np.tile(f_values_plot, (len(x_plot), 1, 1))
        
        # Get predictions
        p_pred = self.predictions([x_plot, a_plot, a_context_repeated, f_context_repeated, f_values_repeated])
        p_pred = p_pred.flatten()
        
        # Create subplots
        fig, ax = plt.subplots(3, 1, figsize=(10, 12))
        
        # Solution p(x)
        ax[0].plot(xdisc, p_pred, 'b-', label='Predicted', linewidth=2)
        ax[0].plot(xdisc, p_true, 'r--', label='True', linewidth=2)
        ax[0].set_title(f'Solution p(x) - Sample {ind}')
        ax[0].set_ylabel('p(x)')
        ax[0].legend()
        ax[0].grid(True)
        
        # Coefficient a(x)
        ax[1].plot(xdisc, a_true, 'g-', label='Coefficient a(x)', linewidth=2)
        ax[1].set_title(f'Coefficient a(x) - Sample {ind}')
        ax[1].set_ylabel('a(x)')
        ax[1].legend()
        ax[1].grid(True)
        
        # Forcing term f(x)
        ax[2].plot(xdisc, f_true, 'm-', label='Forcing f(x)', linewidth=2)
        ax[2].set_title(f'Forcing Term f(x) - Sample {ind}')
        ax[2].set_ylabel('f(x)')
        ax[2].set_xlabel('x')
        ax[2].legend()
        ax[2].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'{log_dir}sample_{ind}_step_{step}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        if wb:
            wandb.log({f"plot_sample_{ind}": wandb.Image(plot_filename)}, step=step)
