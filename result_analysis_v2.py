import os
import pandas as pd
import json
from openpyxl import load_workbook

# Fixed constants
GT_FILE = "./GT.xlsx"

# User-defined parameters
model = "chatgpt-4o-latest"
num_shots = 1
prompt_version = "v7.0"
temperature = "0.7"
input_style = "2row"

# Dynamically generate results directory
results_dir = f"./result_{model}/{num_shots}_shot_{prompt_version}_{temperature}_{input_style}"

# Helper function to sort rep directories numerically
def sort_reps(rep_path):
    """Extract the numeric part of the rep folder name for sorting."""
    rep_number = int(rep_path.split('\\rep')[-1])  # Adjust for '\\' in Windows paths
    return rep_number

# Get list of rep directories and sort them numerically
rep_dirs = sorted(
    [os.path.join(results_dir, d) for d in os.listdir(results_dir) if d.startswith("rep")],
    key=sort_reps
)
num_reps = len(rep_dirs)

# Load ground truth data
gt_table = pd.read_excel(GT_FILE)
gt_labels = dict(zip(gt_table['Sample'], gt_table['Grade']))

# Get case names and expected number of cases
case_names = list(gt_labels.keys())

# Collect thoughts data for the Thoughts Table
thoughts_data = []

# Process each case
for case in case_names:  # Iterate through each case
    for rep, rep_dir in enumerate(rep_dirs, start=1):  # Iterate through each rep folder
        json_file = os.path.join(rep_dir, f"{case}.json")  # Construct JSON file path
        with open(json_file, 'r') as file:
            data = json.load(file)
            thought = data.get("Thoughts", "No thoughts provided")  # Extract thoughts
        # Append data for thoughts table
        thoughts_data.append({"Case Name": case, "Rep": f"Rep {rep}", "Thoughts": thought})

# Create thoughts DataFrame
thoughts_df = pd.DataFrame(thoughts_data)

# Create datagram (case analysis)
datagram = pd.DataFrame({'Case Name': case_names})
datagram['Ground Truth'] = datagram['Case Name'].map(gt_labels)

# Collect rep results for datagram
rep_results = {}
for rep, rep_dir in enumerate(rep_dirs, start=1):
    summary_file = os.path.join(rep_dir, "summary.csv")
    df = pd.read_csv(summary_file)
    rep_results[f'Rep {rep}'] = dict(zip(df['Query ID'], df['T-cell mediated rejection']))

# Add rep results to datagram
for rep in range(1, num_reps + 1):
    datagram[f'Rep {rep}'] = datagram['Case Name'].map(rep_results.get(f'Rep {rep}', {}))

# Calculate correct predictions
datagram['Correct Predictions'] = datagram.apply(
    lambda row: sum(row[f'Rep {rep}'] == row['Ground Truth'] for rep in range(1, num_reps + 1)), axis=1
)

# Calculate over-graded predictions
datagram['Over-Graded'] = datagram.apply(
    lambda row: sum(row[f'Rep {rep}'] > row['Ground Truth'] for rep in range(1, num_reps + 1) if pd.notnull(row[f'Rep {rep}'])), axis=1
)

# Assign difficulty levels
datagram['Level'] = datagram['Correct Predictions'].apply(
    lambda x: 'Easy' if x > 7 else 'Intermediate' if 4 <= x <= 7 else 'Hard'
)

# Save datagram and thoughts to Excel
save_dir  = f"./result_analysis_{model}/{num_shots}_shot_{prompt_version}_{temperature}_{input_style}"
os.makedirs(save_dir,exist_ok=True)
output_file = os.path.join(save_dir, "case_analysis_datagram.xlsx")
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    datagram.to_excel(writer, sheet_name="Case Analysis", index=False)
    thoughts_df.to_excel(writer, sheet_name="Thoughts Table", index=False)

print(f"Case analysis datagram and thoughts table saved to {output_file}")

# Display previews
print("Datagram Preview:")
print(datagram.head())

print("\nThoughts Table Preview:")
print(thoughts_df.head(20))
