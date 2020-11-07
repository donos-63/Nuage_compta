import os
import modules.prepare_pictures as prep_pic
import modules.prepare_model as prep_mod
import modules.tools.file_helper as file_help

GLOBAL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GLOBAL_DIGITS = "0123456789"

def compute_sample():
    file_help.get_data()

    input_folder = os.path.join( file_help.DATA_IN_FOLDER,file_help.ALPHABET_DATASET_FOLDER)
    output_folder = os.path.join( file_help.DATA_CURATED_FOLDER, file_help.DEFAULT_PICTURE_NAME)
    prep_pic.convert_to_training_csv(300, input_folder, output_folder )

    input_folder = os.path.join( file_help.DATA_CURATED_FOLDER, file_help.DEFAULT_PICTURE_NAME)
    output_folder = os.path.join( file_help.DATA_OUT_FOLDER,file_help.MODELS_FOLDER)
    prep_mod.prepare_model( input_folder, output_folder)

compute_sample()
