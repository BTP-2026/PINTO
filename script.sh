#!/bin/bash

# Ask the user which folder they want to use
echo "Enter the Numerical example that you need to run?"
echo "1. Advection"
echo "2. Burgers"
echo "3. Kovasznay Flow"
echo "4. Beltrami Flow"
echo "5. Lid-Driven Cavity Flow"
read -p "Enter the choice as 1, 2, 3, 4, or 5: " choice

echo "Enter the Model type?"
echo "1. PINTO"
echo "2. PI-DeepONets"
read -p "Enter the choice as 1 or 2: " model_choice


# Set the folder path based on user choice
if [ "$choice" = "1" ]; then
  if [ "$model_choice" = "1" ]; then
    main_folder="Advection/Code/PINTO"
    echo "Using PINTO for Advection"
  else
    main_folder="Advection/Code/PI-DeepONet"
    echo "Using PI-DeepONets for Advection"
  fi
elif [ "$choice" = "2" ]; then
  if [ "$model_choice" = "1" ]; then
    main_folder="Burgers/Code/PINTO"
    echo "Using PINTO for Burgers"
  else
    main_folder="Burgers/Code/PI-DeepONet"
    echo "Using PI-DeepONets for Burgers"
  fi
elif [ "$choice" = "3" ]; then
  if [ "$model_choice" = "1" ]; then
    main_folder="Kovasznay/Code/PINTO"
    echo "Using PINTO for Kovasznay Flow"
  else
    main_folder="Kovasznay/Code/PI-DeepONet"
    echo "Using PI-DeepONets for Kovasznay Flow"
  fi
elif [ "$choice" = "4" ]; then
  if [ "$model_choice" = "1" ]; then
    main_folder="Beltrami/Code/PINTO"
    echo "Using PINTO for Beltrami Flow"
  else
    main_folder="Beltrami/Code/PI-DeepONet"
    echo "Using PI-DeepONets for Beltrami Flow"
  fi
elif [ "$choice" = "5" ]; then
  if [ "$model_choice" = "1" ]; then
    main_folder="LidDrivenCavity/Code/PINTO"
    echo "Using PINTO for Lid-Driven Cavity Flow"
  else
    main_folder="LidDrivenCavity/Code/PI-DeepONet"
    echo "Using PI-DeepONets for Lid-Driven Cavity Flow"
  fi
else
    echo "Invalid choice. Exiting."
    exit 1
fi

# Navigate to the chosen folder
cd "$main_folder"

# Run the Python script the folder
if [ "$choice" = "1" ]; then
  if [ "$model_choice" = "1" ]; then
    python3 PINTO_model.py
  else
    python3 DeepONet_model.py
  fi
elif [ "$choice" = "2" ]; then
  if [ "$model_choice" = "1" ]; then
    python3 be1d_model.py
  else
    python3 be1d_DeepONet_model.py
  fi
elif [ "$choice" = "3" ]; then
  if [ "$model_choice" = "1" ]; then
    python3 KF_PINTO_model.py
  else
    python3 KF_DeepONet_model.py
  fi
elif [ "$choice" = "4" ]; then
  if [ "$model_choice" = "1" ]; then
    python3 Bel_PINTO_model.py
  else
    python3 Bel_DeepONet_model.py
  fi
elif [ "$choice" = "5" ]; then
  if [ "$model_choice" = "1" ]; then
    python3 LDF_PINTO_model.py
  else
    python3 LDF_DeepONet_model.py
  fi
fi