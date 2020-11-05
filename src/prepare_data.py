import os
import modules.prepare_pictures as prep_pic
import modules.prepare_model as prep_mod
import modules.tools.file_helper as file_help

GLOBAL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GLOBAL_DIGITS = "0123456789"

def compute_character(character, prefix, characters_referential):
    file_help.get_data()

    characters = [c for c in characters_referential]

    try:
        character_value_index = characters.index(character)
    except:
        print("character not found")
        return

    input_folder = os.path.join( file_help.DATA_IN_FOLDER,file_help.ALPHABET_DATASET_FOLDER, character)
    output_folder = os.path.join( file_help.DATA_OUT_FOLDER,file_help.ALPHABET_DATASET_FOLDER)
    prep_pic.convert_to_csv(character_value_index, prefix, input_folder, output_folder )

    input_folder = os.path.join( file_help.DATA_OUT_FOLDER,file_help.ALPHABET_DATASET_FOLDER, file_help.generate_picture_name(character_value_index, prefix))
    output_folder = os.path.join( file_help.DATA_OUT_FOLDER,file_help.MODELS_FOLDER)
    prep_mod.prepare_model(character_value_index, prefix, input_folder, output_folder)

compute_character('Z', 'a_z', GLOBAL_LETTERS)


'''
done : Z


'''