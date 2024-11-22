import os

def create_unique_folder(base_folder_name):
    # Initialize the folder name
    folder_name = base_folder_name
    count = 1

    # While a folder with this name exists, modify the name
    while os.path.exists(folder_name):
        folder_name = f"{base_folder_name}_{count}"
        count += 1

    # Create the folder
    os.makedirs(folder_name)
    return folder_name