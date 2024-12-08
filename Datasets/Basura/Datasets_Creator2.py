'''
Specify in the constants how many artists you want to consider and how many images per artist 
(ALL if you want them all)
'''
import pandas as pd
import csv
from collections import defaultdict
import os
import shutil


##### CONSTANTS #####
datasets_path = "ALL_IMAGES" # Directory with all the test and train datasets
TOP_ARTISTS = 5 # Specify how many artist will have the new dataset
NUM_IMAGES_PER_ARTIST = 100 # How many pictures for each artists, specifay "ALL" if you want to take all the pictures

##### FUNCTIONS #####
def delete_images_with_no_date(file_csv):    
    """
    Deletes images with null or missing values in the 'date' column from the provided CSV file.

    Parameters:
    - file_csv: Path to the CSV file containing image data.

    Returns:
    - DataFrame: DataFrame with rows filtered out where the 'date' column is null or missing.
    """
    # List to store the images with null or missing values in the 'date' column.
    images_to_delete = []
    
    # Read the CSV file and look for images with null or missing values in the 'date' column.
    with open(file_csv, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            date = row['date']
            if not date or date == 'nan':
                image = row['new_filename']
                images_to_delete.append(image)
                
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_csv)
    
    # Filter out rows where the 'new_filename' column exists in images_to_delete
    df_filtered = df[~df['new_filename'].isin(images_to_delete)]
                    
    # List subfolders in the main folder
    subfolders = os.listdir(datasets_path)

    for subfolder in subfolders:
        subfolder_path = os.path.join(datasets_path, subfolder)
        # Check if the subfolder is a directory
        if os.path.isdir(subfolder_path):
            # List sub-subfolders within each subfolder
            sub_subfolders = os.listdir(subfolder_path)
            for sub_subfolder in sub_subfolders:
                sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder)
                # Check if the sub-subfolder is a directory
                if os.path.isdir(sub_subfolder_path):
                    # List files within each sub-subfolder
                    files = os.listdir(sub_subfolder_path)
                    for image in files:
                        if image in images_to_delete:
                            file_path = os.path.join(sub_subfolder_path, image)
                            os.remove(file_path)

    return df_filtered

                        
                        
def get_info_after_clean(file_csv):
    """
    Reads the CSV file, counts the total number of images before cleaning, and calculates the percentage of images deleted.
    
    Parameters:
    - file_csv: Path to the CSV file containing image data.

    Returns:
    - None
    """
    df = pd.read_csv(file_csv)
    total_images_before_clean = df.shape[0]
    
    # List subfolders in the main folder
    subfolders = os.listdir(datasets_path)
    
    total_files = 0
    for subfolder in subfolders:
        subfolder_path = os.path.join(datasets_path, subfolder)
        # Check if the subfolder is a directory
        if os.path.isdir(subfolder_path):
            # List sub-subfolders within each subfolder
            sub_subfolders = os.listdir(subfolder_path)
            for sub_subfolder in sub_subfolders:
                sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder)
                # Check if the sub-subfolder is a directory
                if os.path.isdir(sub_subfolder_path):
                    # List files within each sub-subfolder
                    files = os.listdir(sub_subfolder_path)
                    total_files += len(files)
                    
    print("Percentage of images deleted: " + str((1 - (total_files / total_images_before_clean)) * 100) + "%")
    print("Total images now:", total_files)
    

def get_top_n_artists(df, n = TOP_ARTISTS, NUM_IMAGES_PER_ARTIST = NUM_IMAGES_PER_ARTIST):
    """
    Returns the rows of the top n artists that appear most frequently in the given DataFrame.

    Parameters:
    - df: DataFrame containing the data after cleaning.
    - n: Number of top artists to select.
    - NUM_IMAGES_PER_ARTIST: Number of images per artist to select. If None, select all images per artist.

    Returns:
    - DataFrame: Rows of the top n artists.
    """
    # Count the occurrences of each artist
    artist_counts = df['artist'].value_counts()
    # Get the top n artists
    top_artists = artist_counts.head(n).index.tolist()
    
    if NUM_IMAGES_PER_ARTIST == "ALL":
        # Filter the DataFrame to include only rows with the top n artists
        top_artists_rows = df[df['artist'].isin(top_artists)]
    else:
        top_artists_rows = pd.DataFrame()
        for artist in top_artists:
            artist_rows = df[df['artist'] == artist]
            top_artist_images = artist_rows.head(NUM_IMAGES_PER_ARTIST)
            top_artists_rows = pd.concat([top_artists_rows, top_artist_images])
            
    return top_artists_rows

def copy_images_to_destination_directory(destination_directory, image_filenames):
    """
    Copies image files from the source directory to the destination directory.

    Parameters:
    - destination_directory: The directory where the images will be copied.
    - image_filenames: List of image filenames to copy.

    Returns:
    - None
    """
    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
        print(f"Directory '{destination_directory}' created successfully.")

    # Iterate over each image filename
    for image_filename in image_filenames:
        # Iterate over each directory in the source directory
        for directory, _, files in os.walk(datasets_path):
            # Check if the image file is in the list of files in the current directory
            if image_filename in files:
                # Construct the full path of the source file
                source_path = os.path.join(directory, image_filename)
                # Construct the full path of the destination file
                destination_path = os.path.join(destination_directory, image_filename)
                
                # Check if the file already exists in the destination directory
                if not os.path.exists(destination_path):
                    # Copy the file to the destination directory only if it doesn't exist
                    shutil.copy(source_path, destination_path)
                    print(f"File '{image_filename}' copied successfully.")
                else:
                    print(f"File '{image_filename}' already exists in the destination directory.")
                
                # Break the loop once the file is found
                break
        else:
            print(f"File '{image_filename}' not found in the source directories.")

    print("Process completed.")

    
    
##### MAIN #####

# Delete images that have no date registered
df_filtered = delete_images_with_no_date("all_data_info.csv")

# Get info after cleaning the data
get_info_after_clean("all_data_info.csv")

# Get the dataset with the top artists
df_top_artists = get_top_n_artists(df_filtered, TOP_ARTISTS, NUM_IMAGES_PER_ARTIST)

# Save the dataset
df_top_artists.to_csv('TOP{TOP_ARTISTS}_ARTISTS_WITH_{NUM_IMAGES_PER_ARTIST}_PICTURES.csv', index=False)

# Create the new directory
print("\n\n")
destination_directory = f"TOP{TOP_ARTISTS}_ARTISTS_WITH_{NUM_IMAGES_PER_ARTIST}_PICTURES"
copy_images_to_destination_directory(destination_directory, df_top_artists["new_filename"].tolist())




