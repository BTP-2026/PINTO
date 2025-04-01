Necessary packages in the python environment:
   python version: >= 3.10.
   create the virtual environment using the following command:
        - python3 -m venv pinto
        - source ~/pinto/bin/activate
        - pip install -r requirements.txt

Documentation to run the code:
    -Each Numerical Example folder consists of three subfolders Code, Post_Processing, and Trained_models
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