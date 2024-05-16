import os

class FileHandler:
    def __init__(self, project):
        self.project = project

    def get_all_files(self, directory):
        """
        Recursively gets all files within a directory.
        """
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_paths.append(os.path.join(root, file))
        return file_paths

    def read_file(self, file_path):
        """
        Reads the content of a file.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()