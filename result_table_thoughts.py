import os
import json
import pandas as pd

# Define the root directory where the results are stored
k = 1
pv = 'v7.0'
temperature = 0.7
res_dir = 'result_4o'
task = f'{k}_shot_{pv}_{temperature}'
root_dir = f'{res_dir}/{task}'

# List of all samples and repetitions
samples = [f + '.json' for f in os.listdir("./data")]
repetitions = [f"rep{i}" for i in range(1, 11)]

# Load the ground truth data
gt_file_path = './data/GT.xlsx'
gt_df = pd.read_excel(gt_file_path)

# Prepare an empty list to store extracted data
data = []

# Loop through each sample and repetition
for sample in samples:
    for rep in repetitions:
        # Build the file path
        file_path = os.path.join(root_dir, rep, sample)
        
        # Check if the file exists
        if os.path.exists(file_path):
            # Open and load the JSON file
            with open(file_path, 'r') as file:
                result = json.load(file)
            
            # Extract the relevant information and store it in a dictionary
            entry = {
                "Sample": sample,
                "Repetition": rep,
                "T-cell mediated rejection": result.get("T-cell mediated rejection"),
                "Thoughts": result.get("Thoughts")
            }
            
            # Append the entry to the data list
            data.append(entry)

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Merge with ground truth data based on the Sample column
df = df.merge(gt_df, on="Sample", how="left")

# Save the DataFrame to an Excel file
output_path = "heart_transplant_results_with_gt.xlsx"
df.to_excel(output_path, index=False)

print(f"Results saved to {output_path}")
