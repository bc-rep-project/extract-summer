import os
import shutil

def extract_contents(directory):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    new_directory = os.path.join(script_dir, 'Extracted', os.path.basename(directory))
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(dirpath, directory)
            new_dirpath = os.path.join(new_directory, relative_path)
            os.makedirs(new_dirpath, exist_ok=True)
            new_filename = os.path.splitext(filename)[0] + '.txt'
            new_filepath = os.path.join(new_dirpath, new_filename)
            if not os.path.exists(new_filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        contents = file.read()
                    with open(new_filepath, 'w', encoding='utf-8') as new_file:
                        new_file.write(contents)
                except UnicodeDecodeError:
                    print(f"Could not decode file: {filepath}")

# Replace 'your_directory' with the path to the directory you want to extract contents from
extract_contents('/workspaces/extract-summer/aider')