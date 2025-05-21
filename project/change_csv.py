import os
import glob
import pandas as pd

def update_csv_filenames(csv_path, train_total_folder, output_csv_path):
    # Load your CSV
    df = pd.read_csv(csv_path)

    # Build a lookup from prefix to full filename
    image_files = glob.glob(os.path.join(train_total_folder, "*.jpg"))
    prefix_to_filename = {}
    for path in image_files:
        basename = os.path.basename(path)
        if '_JPG' in basename:
            prefix = basename.split('_JPG')[0] + '_JPG'
            if prefix not in prefix_to_filename:  # First match wins
                prefix_to_filename[prefix] = basename

    # Update filenames in the CSV
    updated_filenames = []
    not_found = 0
    for original_name in df['filename']:
        prefix = original_name.split('_JPG')[0] + '_JPG'
        new_name = prefix_to_filename.get(prefix)
        if new_name:
            updated_filenames.append(new_name)
        else:
            updated_filenames.append(original_name)  # Keep old name
            not_found += 1

    df['filename'] = updated_filenames
    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to: {output_csv_path}")
    print(f"Could not find matches for {not_found} filenames.")

# Example usage:
update_csv_filenames(
    csv_path=r'data\_annotations_objects.csv',
    train_total_folder=r'images\train_total',
    output_csv_path=r'data\updated.csv'
)
