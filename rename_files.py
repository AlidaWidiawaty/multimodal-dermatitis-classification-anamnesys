import os
import re


def rename_files_in_folder(folder_path):
    # Get a list of all files in the directory
    files = os.listdir(folder_path)

    for filename in files:
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)

        # Check if it's a file (not a subdirectory)
        if os.path.isfile(file_path):
            # Remove "_jpeg_" or "_jpg_" using regular expressions
            new_filename = re.sub(r"_(jpeg|jpg|JPG|JPEG)_", "_", filename)
            new_filename = re.sub("-2", "", new_filename)
            # new_filename = re.sub("-_", "_", new_filename)

            # Only rename the file if its name has changed
            if filename != new_filename:
                new_file_path = os.path.join(folder_path, new_filename)

                if os.path.exists(new_file_path):
                    print(f"⚠️ File already exists: {new_filename}")
                    continue

                os.rename(file_path, new_file_path)
                print(f"Renamed: {filename} -> {new_filename}")


# Specify the folder path
folder_path = "./dataset/processed"

# Call the function to rename files
rename_files_in_folder(folder_path)