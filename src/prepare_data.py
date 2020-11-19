import os
import modules.prepare_pictures as prep_pic
import modules.prepare_model as prep_mod
import modules.tools.file_helper as file_help
import modules.apply_model as analysis

characters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
#characters = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


def compute_sample():
    file_help.get_data()

    input_folder = os.path.join( file_help.DATA_IN_FOLDER,file_help.ALPHABET_DATASET_FOLDER)
    output_folder = os.path.join( file_help.DATA_CURATED_FOLDER, file_help.DEFAULT_PICTURE_NAME)
    prep_pic.convert_to_training_csv(15000, input_folder, output_folder )

    input_folder = os.path.join( file_help.DATA_CURATED_FOLDER, file_help.DEFAULT_PICTURE_NAME)
    output_folder = os.path.join( file_help.DATA_OUT_FOLDER,file_help.MODELS_FOLDER)
    prep_mod.prepare_model2( input_folder, output_folder, characters)

    analysis.analyse_picture2('C:\\prairie\\projet9\\Nuage_compta\\data\\in\\formation-data_ia_test.jpeg', characters)

compute_sample()
