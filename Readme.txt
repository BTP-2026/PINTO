Necessary packages in the python environment:
   python version: >= 3.10.
   create the virtual environment using the following command:
        - python3 -m venv pinto
        - source ~/pinto/bin/activate
        - pip install -r requirements.txt

Detailed Documentation:
    -Each Numerical Example folder consists of three subfolders: Code, Post_Processing, and Trained_models
    -About Folders:
        Numerical examples/
        ├── Code
        │   ├── PINTO
        │   │   ├── model file, Pde file, utils
        │   ├── DeepONets
        │   │   ├── model file, Pde file, utils
        ├── Post_processing
        │   ├── Post_Processing.ipynb - contains the code for post-processing and plots in paper of corresponding test cases
        ├── Trained models - have the trained models of PINTO and DeepONets for corresponding test cases

   - activate the "pinto" venv
   - navigate the code folder of corresponding numerical example and

   data folder for burgers
   https://indianinstituteofscience-my.sharepoint.com/:u:/g/personal/ksumanth_iisc_ac_in/EWeItMUTYulKit8tqgKnh44BzzUDxc-whoJadi2QjGLBuA?e=IYR3p0

   In a Code folder, you will find the three files Pde.py, utils.py, and model.py:
     PdeModel class documentation: (EquationType_Pde.py file in PINTO/DeepONets folder for each Numerical example)
        -Attributes
            - inputs: domain/boundary/initial coordinates for QPE units,
                      initial/boundary coordinates sequence for BPE units, initial/boundary conditions sequence for BVE units
            - outputs: scalar/vector quantities of our interest
            - model: PINTO/DeepONet model object (built using Tensorflow Functional API)
            - optimizer: optimizer object
            - loss_fn: loss function object
            - parameters: Governing equations parameters (Re, nue, beta, etc.)
            - inner/bound/init_data: Tensorflow dataset object
            - batches
            - loss_tracker: metrics to track the various loss terms during training

        - Methods
            - __init__: constructor to initialize the attributes
            - create_data_pipeline: to create the tensorflow dataset for training/validation
            - Pde_residual: to compute the PDE residuals of corresponding numerical example using Automatic Differentiation
            - train_step: to perform a single training step (which essentially consists of four major steps)
                - Forward pass
                - Compute loss
                - Backward pass (computing gradients using back propagation)
                - Update weights (applying gradients to the model weights using the optimizer)
            - test_step: to perform a single test step (which essentially consists of two major steps)
                - Forward pass
                - Compute loss
            -reset_metrics: to reset the loss metrics at beginning of each epoch
            -get_model_graph: to get the PINTO/DeepONet model plot
            -run: performs the training of the model for provided number of epochs
            -prediction: to perform the prediction of the model for given input coordinates
            -get_plots: to get the plots of the model predictions at regular intervals (plot_freq in run method defines the interval to plot)

     model.py
