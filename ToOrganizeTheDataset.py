import os
import pandas as pd
import shutil

# Load the CSV file
csv_path = "UnorganisedDataset/fitz_undersampled_test_final.csv"  # Update this with the path to your CSV file
image_folder = "UnorganisedDataset/val"  # Update this with the folder containing your images
output_folder = "UnorganisedDataset/val"

df = pd.read_csv(csv_path)

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through rows in the CSV
for _, row in df.iterrows():
    file_name = row['file']
    phototype = row['phototype']

    # Handle cases like "I & II" by splitting into multiple types
    skin_types = [t.strip() for t in phototype.split('&')]

    for skin_type in skin_types:
        skin_type_folder = os.path.join(output_folder, skin_type)

        # Create folder for skin type if it doesn't exist
        if not os.path.exists(skin_type_folder):
            os.makedirs(skin_type_folder)

        # Move or copy the image to the respective folder
        src_path = os.path.join(image_folder, file_name)
        dst_path = os.path.join(skin_type_folder, file_name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: {file_name} not found in {image_folder}.")

print("Dataset preprocessing complete.")
