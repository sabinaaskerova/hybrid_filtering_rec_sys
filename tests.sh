#!/bin/bash tests.sh or bash tests.sh #################

# This script is used to 
# 1) train models on train set 
# 2) evaluate the performance of the four models on the test and train+test datasets.

# Define the models and file names
models=("gd_mf" "hybrid_mf" "mlp_mf" "pytorch_gd_mf")
files=("ratings_test.npy" "ratings_train.npy" "ratings_train_test.npy")

# Loop through each model and file name
for model in "${models[@]}"; do
    for file in "${files[@]}"; do
        # Execute the training command and measure the time it takes
        start_time=$(date +%s)
        python generate_custom.py --model "$model" --name "$file"
        end_time=$(date +%s)
        elapsed_time=$((end_time - start_time))
        
        # Print the model, file name, and elapsed time
        echo "Model: $model" >> evaluation_output.txt
        echo "File: $file" >> evaluation_output.txt
        echo "Elapsed Time: $elapsed_time seconds" >> evaluation_output.txt
        
        # Execute the evaluation command and save the output to a file
        python evaluate_models.py --name "$file" >> evaluation_output.txt
        echo "--------------" >> evaluation_output.txt
    done
done