import os
import pandas as pd

# Paths to your two image folders
folder1 = r"C:\Users\melis\Master2_programme\_Image_analysis\iapr2025\project\images\train"
folder2 = r"C:\Users\melis\Master2_programme\_Image_analysis\iapr2025\project\images\valid"

# Path to the original CSV file
input_csv = r"C:\Users\melis\Master2_programme\_Image_analysis\iapr2025\project\data\_annotations.csv"

# Output CSV filenames
output_csv1 = "train.csv"
output_csv2 = "valid.csv"

# Load CSV into DataFrame
df = pd.read_csv(input_csv)

# Get sets of filenames in each folder for fast lookup
files_folder1 = set(os.listdir(folder1))
files_folder2 = set(os.listdir(folder2))

# Define a function to decide which folder a filename belongs to
def assign_folder(filename):
    if filename in files_folder1:
        return 1
    elif filename in files_folder2:
        return 2
    else:
        return 0  # not found in either folder

# Apply the function to create a new column
df["folder"] = df["filename"].apply(assign_folder)

# Split DataFrame based on folder
df_folder1 = df[df["folder"] == 1].drop(columns=["folder"])
df_folder2 = df[df["folder"] == 2].drop(columns=["folder"])
df_not_found = df[df["folder"] == 0]

# Save the splits
df_folder1.to_csv(output_csv1, index=False)
df_folder2.to_csv(output_csv2, index=False)

print(f"Saved {len(df_folder1)} rows to {output_csv1}")
print(f"Saved {len(df_folder2)} rows to {output_csv2}")

if not df_not_found.empty:
    print(f"Warning: {len(df_not_found)} rows with filenames not found in either folder.")
