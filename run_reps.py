import subprocess
from concurrent.futures import ThreadPoolExecutor
import time

# Define the parameters you want to use
model = "chatgpt-4o-latest"
max_tokens = 300
detail = "high"
batch = 5
pv = "v7.0"
thought = True  # Boolean flag

# Options for temperature, k, and data
temperature_options = [
    0.3, 
    0.5, 
    # 0.7, 
    1.0
]

k_options = [
    # 0,
    1,
    # 2,
    # 3
]

data_options = [
    # 'individual',
    # 'individual_enlarged',
    # '1row',
    # '1row_enlarged',
    '2row',
    # '2row_enlarged'
]

# Function to run a single rep for given k, data, and temperature
def run_rep(rep, k, data, temperature):
    try:
        print(f"Running Rep {rep} for k={k}, data={data}, temperature={temperature}...")
        
        # Construct the command
        command = [
            "python", "./rejection_grade_prediction.py",
            "--model", model,
            "--max_tokens", str(max_tokens),
            "--temperature", str(temperature),
            "--detail", detail,
            "--batch", str(batch),
            "--k", str(k),
            "--rep", str(rep),
            "--pv", pv,
            "--data", data,
        ]
        # Add the --thought flag if thought is True
        if thought:
            command.append("--thought")
        
        # Run the command
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"Rep {rep} completed for k={k}, data={data}, temperature={temperature}.")
        print(result.stdout)  # Optionally log the output
    except subprocess.CalledProcessError as e:
        print(f"Error in Rep {rep} for k={k}, data={data}, temperature={temperature}: {e.stderr}")

# Main loop to run all reps in parallel for each k, data, and temperature
def run_all_reps_parallel(k, data, temperature, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_rep, rep, k, data, temperature) for rep in range(1, 11)
        ]
        for future in futures:
            try:
                future.result()  # Wait for completion and raise exceptions if any
            except Exception as e:
                print(f"Error in parallel execution: {e}")

# Main execution loop
for temperature in temperature_options:
    for k in k_options:
        for data in data_options:
            print(f"Running all 10 reps in parallel for k={k}, data={data}, temperature={temperature}...")
            run_all_reps_parallel(k, data, temperature)
