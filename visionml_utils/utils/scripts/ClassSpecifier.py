import os
import re

class ClassSpecifier():
    def specify_class(self, folder_path, new_class):
        """
        Modifies the class identifier on the first line of each text file to the new_class.

        Args:
        folder_path (str): Path to the folder containing the text files.
        new_class (int): The new number to set as the first number on the first line of each file.
        """
        # Make sure new_class is a string
        new_class_str = str(new_class)

        # Regex pattern to match the first number at the beginning of each line
        pattern = re.compile(r'^\d+', re.MULTILINE)

        # Iterate through each file in the specified folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # Check if the file is a text file
            if os.path.isfile(file_path) and filename.endswith('.txt'):
                # Read the content of the file
                with open(file_path, 'r') as file:
                    content = file.read()
                
                # Replace the first number on each line with the new number
                modified_content = pattern.sub(new_class_str, content)
                
                # Write the modified content back to the file
                with open(file_path, 'w') as file:
                    file.write(modified_content)

def main():
    specifier = ClassSpecifier()
    # Example usage
    folder_path_train = r'C:\Users\User\Box\RoboCup\RoboCup 2024\Teams\Vision\Data\soccer_ball_finetune\train\labels'
    folder_path_valid = r'C:\Users\User\Box\RoboCup\RoboCup 2024\Teams\Vision\Data\soccer_ball_finetune\valid\labels'
    new_class = 0  # The new number to replace with
    specifier.specify_class(folder_path_train, new_class)
    specifier.specify_class(folder_path_valid, new_class)

if __name__ == '__main__':
    main()