#!/usr/bin/env bash
echo "Setting environment variables..."

# Define the default path
source .env
if [ $? -ne 0 ]; then
  echo "Error: .env file not found or could not be sourced."
  exit 1
fi

DEFAULT_PATH="./datasets"

# Check if the BASE_DIR environment variable is not set
if [ -z "${BASE_DIR}" ]; then
  echo "The BASE_DIR environment variable is not set."
  read -p "Please enter the path for BASE_DIR [default: ${DEFAULT_PATH}]: " user_input

  # If the user input is empty, use the default path
  if [ -z "${user_input}" ]; then
    BASE_DIR="${DEFAULT_PATH}"
    echo "No input provided. Using default path."
  else
    # Otherwise, use the path provided by the user
    BASE_DIR="${user_input}"
  fi
  echo $BASE_DIR >> .env
fi

echo "Running all notebooks..."
# Below are the primary notebooks used to generate figures for the paper.
# Other complementary notebooks are available in the notebooks/ directory.
jupyter nbconvert --to notebook --execute --inplace notebooks/fig3-5_view_six_examples.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/fig4_compare_synth_real_distributions.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/fig6_kV_mA_variation.ipynb # uses EXPERIMENT_NAME for kVp/mA variation
jupyter nbconvert --to notebook --execute --inplace notebooks/fig7_create_multimodel_ROC.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/fig8_compare_false_negatives.ipynb

echo "Done."
