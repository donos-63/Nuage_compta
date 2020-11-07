import os
import sys
from os import walk
from PIL import Image
import numpy as np
import random

src_dir = os.path.join(os.getcwd(), 'src')
sys.path.insert(0,src_dir) 

import modules.tools.file_helper as file_help

DEFAULT_NAME = "handwritten_data.csv"
GLOBAL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GLOBAL_DIGITS = "0123456789"

def convert_picture_to_str(identifier, pictures_path):
    img = Image.open(pictures_path)
    width, height = img.size

    data = list(img.getdata()) 
    # convert image data to a list of integers
    data = [data[offset:offset+width] for offset in range(0, width*height, width)]

    #transfomr to numpy matrix then to flat data
    matrice = np.array(data, dtype="str")
    picture_to_pixel = str.format("%s%s\n" % (identifier, ','+ ','.join(matrice.flatten())))

    return picture_to_pixel

def generate_identifier(dir_name):
    identifier_str = os.path.basename(dir_name)
    labels = GLOBAL_DIGITS
    labels += GLOBAL_LETTERS
    characters = [c for c in labels]

    try:
        character_value_index = characters.index(identifier_str)
    except:
        print("character not found")
        return -1

    return character_value_index

def convert_to_training_csv(nb_sample, pictures_path, output_file):
    
    resume_file = os.path.join(file_help.DATA_CURATED_FOLDER, 'resume_file.csv')
    if os.path.exists(output_file):
        print('sample already build')
        return

    with open(output_file, 'w') as f:
        with open(resume_file, 'a') as f_resume:
            subfolders = [ f.path for f in os.scandir(pictures_path) if f.is_dir() ]
            for subfolder in subfolders:
                i=0
                files = os.listdir(subfolder)
            
                if(nb_sample > len(files)):
                    nb_sample = len(files)

                sample_dataset = random.sample(set(files), nb_sample)
                identifier = generate_identifier(subfolder)
                for sample in sample_dataset:
                    filename = os.path.join(subfolder, sample)
                    row = convert_picture_to_str(identifier, filename)

                    #display progress bar
                    i += 1
                    done = int(100 * i / nb_sample)
                    sys.stdout.write("\r%s : [%s%s] %s/%s" % (os.path.basename(subfolder), '=' * done, ' ' * (100-done), i, nb_sample)) 
                    f.write(row)
                    f_resume.write(sample + '\n')

                print("")
        
        print("Generation completed")

if __name__ == "__main__":
    #test with 'Z'
    convert_to_training_csv(100, 'C:\\prairie\\projet9\\Nuage_compta\\data\\in\\alphabet-dataset', 'C:\\prairie\\projet9\\Nuage_compta\\data\\curated\\handwritten_data.csv' )

