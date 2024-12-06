import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
model = "chatgpt-4o-latest_new"  # chatgpt-4o-latest
results_dir = f"./result_{model}"
gt_file = "./GT.xlsx"
save_dir = f"./result_tables_{model}"
temp = [
    '0.0', 
    '0.3', 
    '0.5', 
    '0.7', 
    '1.0'
]

# Parameters for analysis
k_values = [
    # 0, 
    1, 
    # 2, 
    # 3
]
data_types = [
    # 'individual', 
    # 'individual_enlarged', 
    # '1row',
    # '1row_enlarged', 
    '2row',
    '2row_noThought', 
    # '2row_enlarged'
]
reps = 10

# Expected number of cases
expected_cases = 16

# Ensure save directory exists
os.makedirs(save_dir, exist_ok=True)

# Load ground truth labels
gt_table = pd.read_excel(gt_file)
gt_labels = dict(zip(gt_table['Sample'], gt_table['Grade']))  # Mapping of sample to true grade

def load_results(rep_dir):
    """Load results from a single rep directory and ensure all cases are present."""
    try:
        summary_file = os.path.join(rep_dir, 'summary.csv')
        df = pd.read_csv(summary_file)
        
        # Check that all 16 cases are present
        if len(df) != expected_cases:
            missing_cases = set(gt_labels.keys()) - set(df['Query ID'])
            print(f"Warning: Missing cases in {summary_file}: {missing_cases}")
            return pd.DataFrame()  # Return empty DataFrame if cases are missing
        
        df['TrueLabel'] = df['Query ID'].map(gt_labels)  # Add true labels from GT table
        return df
    except FileNotFoundError:
        print(f"Summary file not found in {rep_dir}")
        return pd.DataFrame()  # Return empty DataFrame if file is missing

# Initialize a comprehensive results DataFrame to store results for all parameter combinations
comprehensive_df = pd.DataFrame()

# Initialize list to store results for confusion matrix plotting
all_results = []

for k in k_values:
    for data_type in data_types:
        for t in temp:  # Loop through each temp value
            # Initialize data structure for results
            accuracies = []
            all_preds = []

            # Process each rep
            for rep in range(1, reps + 1):
                rep_dir = os.path.join(results_dir, f"{k}_shot_v7.0_{t}_{data_type}", f"rep{rep}")
                df = load_results(rep_dir)
                
                if not df.empty:
                    # Calculate accuracy for the rep
                    y_true = df['TrueLabel']
                    y_pred = df['T-cell mediated rejection']
                    accuracy = accuracy_score(y_true, y_pred)
                    accuracies.append(accuracy)
                    all_preds.append(y_pred)
                else:
                    accuracies.append(None)  # Mark missing reps or incomplete cases with None

            # Create DataFrame for individual rep accuracies
            accuracy_df = pd.DataFrame({
                'Rep': list(range(1, reps + 1)),
                'Accuracy': accuracies
            })
            
            # Calculate consensus prediction for each sample (majority vote)
            if all_preds:
                all_preds = np.array(all_preds).T  # Transpose to align predictions for each sample
                consensus_preds = [
                    pd.Series(row).dropna().astype(int).mode()[0] if not pd.Series(row).dropna().empty else None
                    for row in all_preds
                ]  # Majority vote ignoring None
                consensus_accuracy = accuracy_score(y_true, consensus_preds)  # Consensus accuracy
            else:
                consensus_accuracy = None
                consensus_preds = []  # Empty list if no consensus

            # Append consensus accuracy to the accuracy table
            consensus_row = pd.DataFrame({'Rep': ['Consensus'], 'Accuracy': [consensus_accuracy]})
            accuracy_df = pd.concat([accuracy_df, consensus_row], ignore_index=True)

            # Save individual accuracy table for this parameter combination
            table_filename = os.path.join(save_dir, f"{k}_shot_v7.0_{t}_{data_type}.xlsx")
            accuracy_df.to_excel(table_filename, index=False, sheet_name="Accuracy per Rep")

            # Add this parameter combination's results to the comprehensive table
            col_prefix = f"{k}_shot_{data_type}_{t}"
            accuracy_df = accuracy_df.set_index('Rep')
            comprehensive_df = pd.concat([comprehensive_df, accuracy_df.rename(columns={'Accuracy': col_prefix})], axis=1)

            # Store results for confusion matrix plotting later
            all_results.append({
                'K': k,
                'Data Type': data_type,
                'Temp': t,
                'Consensus Accuracy': consensus_accuracy,
                'Consensus Predictions': consensus_preds
            })

            print(f"Processed k={k}, data_type={data_type}, temp={t}")

# Save the comprehensive table to an Excel file
comprehensive_table_filename = os.path.join(save_dir, "comprehensive_accuracy_table.xlsx")
comprehensive_df.to_excel(comprehensive_table_filename)

# Plotting and saving function for 4x4 confusion matrix
def plot_and_save_confusion_matrix(y_true, y_pred, title, filename):
    """Plot and save a 4x4 confusion matrix with labels from 0 to 3."""
    labels = [0, 1, 2, 3]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    
    # Save the confusion matrix plot
    plt.savefig(filename)
    plt.close()  # Close the plot to free memory

# Generate and save confusion matrices for each parameter combination
for result in all_results:
    k = result['K']
    data_type = result['Data Type']
    t = result['Temp']
    consensus_accuracy = result['Consensus Accuracy']
    consensus_preds = result['Consensus Predictions']

    print(f"Results for k={k}, data_type={data_type}, temp={t}")
    print(f"Consensus Accuracy: {consensus_accuracy}")

    # Plot confusion matrix for consensus predictions if available
    if consensus_accuracy is not None:
        # Use true labels corresponding to the samples in consensus predictions
        sample_names = gt_table['Sample']  # List of sample names from GT
        y_true = [gt_labels[sample] for sample in sample_names if sample in gt_labels]
        y_pred = [pred for pred in consensus_preds if pred is not None]  # Filter out None predictions
        
        # Filename for the confusion matrix image
        cm_filename = os.path.join(save_dir, f"confusion_matrix_{k}_shot_{data_type}_{t}.png")
        
        # Plotting and saving
        plot_and_save_confusion_matrix(y_true, y_pred, title=f"Confusion Matrix (k={k}, data={data_type}, temp={t})", filename=cm_filename)