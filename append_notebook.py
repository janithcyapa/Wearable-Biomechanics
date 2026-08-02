import nbformat as nbf

# Read the existing notebook
with open('Joint_angle_and_ekf_estimation.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

# Create the new cells
text4 = """\
## Batch Process All Experiments
Execute the cell below to list all existing experiments in the dataset store and process them one by one. 
Errors in individual experiments will be caught so the loop continues.
"""

code_batch = """\
# @title Process All Experiments
df_experiments = store.list_entries("experiments")

if not df_experiments.empty:
    print(f"Found {len(df_experiments)} experiments. Starting batch processing...\\n")
    for exp_id in df_experiments['entry_id']:
        try:
            process_experiment(exp_id)
            print("-" * 50)
        except Exception as e:
            print(f"❌ Error processing {exp_id}: {e}")
            print("-" * 50)
    print("Batch processing complete! 🎉")
else:
    print("No experiments found in the store.")
"""

# Append the new cells
nb.cells.extend([
    nbf.v4.new_markdown_cell(text4),
    nbf.v4.new_code_cell(code_batch)
])

# Save the notebook back
with open('Joint_angle_and_ekf_estimation.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Appended batch processing cells to the notebook successfully.")
