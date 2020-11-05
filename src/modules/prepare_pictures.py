import os
import sys
from os import walk
from PIL import Image
import numpy as np

src_dir = os.path.join(os.getcwd(), 'src')
sys.path.insert(0,src_dir) 

import modules.tools.file_helper as file_help

DEFAULT_NAME = "handwritten_data.csv"

def convert_to_csv(identifier, prefix, pictures_path, output_path):
    
    output_file = file_help.generate_picture_name (identifier, prefix)
    with open(os.path.join(output_path, output_file), 'w') as f:
        for (_, _, filenames) in walk(pictures_path):
            i=0
            for file in filenames:
                img = Image.open(os.path.join(pictures_path,file))
                width, height = img.size

                data = list(img.getdata()) 
                # convert image data to a list of integers
                data = [data[offset:offset+width] for offset in range(0, width*height, width)]

                #transfomr to numpy matrix then to flat data
                matrice = np.array(data, dtype="str")
                picture_to_pixel = str.format("%s%s\n" % (identifier, ','+ ','.join(matrice.flatten())))

                #add to referential
                f.write(picture_to_pixel)

                #notice progression of the transformation
                #todo : factorize
                i += 1
                done = int(100 * i / len(filenames))
                sys.stdout.write("\r[%s%s] %s/%s" % ('=' * done, ' ' * (100-done), i, len(filenames))) 
        print("Generation completed")


if __name__ == "__main__":
    #test with 'Z'
    convert_to_csv(25, "a_z", 'C:\\prairie\\projet9\\Nuage_compta\\data\\in\\alphabet-dataset\\Z', 'C:\\prairie\\projet9\\Nuage_compta\\data\\out\\alphabet-dataset\\' )
