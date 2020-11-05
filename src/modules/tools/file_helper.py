import os
import sys
import requests
from hurry.filesize import size
import zipfile

def get_root():
    path = os.getcwd()    
    root = False
    i = 0

    while not root:
        if not os.path.exists(os.path.join(path, 'requirements.txt')):
            path = os.path.abspath(os.path.join(path, os.pardir))
        else:
            root = True
        i = i+1

    return path
    
cwd = get_root()
DATASOURCE_PATH = r'https://stdatalake006.blob.core.windows.net/public/alphabet-dataset.zip'
DATA_IN_FOLDER = os.path.join(cwd, 'data', 'in')
DATA_CURATED_FOLDER = os.path.join(cwd, 'data', 'curated')
DATA_OUT_FOLDER = os.path.join(cwd, 'data', 'out')
ALPHABET_DATASET_FOLDER = "alphabet-dataset"
MODELS_FOLDER = "models"
DEFAULT_PICTURE_NAME = "handwritten_data.csv"
DEFAULT_MODEL_NAME = "handwriting.model"
DEFAULT_PLOT_NAME = "plot.png"
DOWNLOADED_FILE_NAME = r'alphabet.zip'

DATASOURCE_DEST = os.path.join(DATA_IN_FOLDER, DOWNLOADED_FILE_NAME)

def generate_picture_name(identifier, prefix):
    return __generate_standard_file_name(identifier, prefix, DEFAULT_PICTURE_NAME)

def generate_model_name(identifier, prefix):
    return __generate_standard_file_name(identifier, prefix, DEFAULT_MODEL_NAME)

def generate_plot_name(identifier, prefix):
    return __generate_standard_file_name(identifier, prefix, DEFAULT_PLOT_NAME)

def __generate_standard_file_name(identifier, prefix, extension):
    return str.format("%s_%s_%s" % (identifier, prefix, extension))


def __download_data():
    if os.path.exists(os.path.join(DATA_IN_FOLDER, DOWNLOADED_FILE_NAME)):
        print("dataset zip already downloaded")
        return

    print('Import %s' % DATASOURCE_PATH)
    with open(DATASOURCE_DEST, "wb") as f:
        response = requests.get(DATASOURCE_PATH, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None: # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            print("Downloading to %s" % (DATASOURCE_DEST))
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(100 * dl / total_length)
                sys.stdout.write("\r[%s%s] %s/%s" % ('=' * done, ' ' * (100-done), size(dl), size(total_length))) 
                sys.stdout.flush()

            print('', end = "\r\n")
            print('Data downloaded')

def __unzip_dataset():

    if(os.path.exists(os.path.join(DATA_IN_FOLDER, ALPHABET_DATASET_FOLDER))
        and os.listdir(os.path.join(DATA_IN_FOLDER, ALPHABET_DATASET_FOLDER))):
        print("output folder already exists")
        return

    print('Unzip %s' % DATASOURCE_DEST)
    if os.path.exists(DATASOURCE_DEST):
        zf = zipfile.ZipFile(DATASOURCE_DEST)
        uncompress_size = sum((file.file_size for file in zf.infolist()))
        extracted_size = 0

        nb_files = len(zf.infolist())
        file_index = 0
        for file in zf.infolist():
            zf.extract(file, DATA_IN_FOLDER)
            
            file_index += 1
            done = int(100 * file_index / nb_files)
            sys.stdout.write("\r[%s%s] %s/%s" % ('=' * done, ' ' * (100-done), file_index, nb_files)) 
            sys.stdout.flush()
        
        del(zf)
        print('Data extracted with success')
    else:
        print("The datasource does not exist") 

def __cleanup_data():
       
    print('Remove %s' % DATASOURCE_DEST)
    if os.path.exists(DATASOURCE_DEST):
        os.remove(DATASOURCE_DEST)
        print("Datasource file cleaned up") 
    else:
        print("The datasource does not exist") 


def get_data():
    if(os.path.exists(os.path.join(DATA_IN_FOLDER, DOWNLOADED_FILE_NAME)) and os.path.exists(os.path.join(DATA_IN_FOLDER, ALPHABET_DATASET_FOLDER))
        and os.listdir(os.path.join(DATA_IN_FOLDER, ALPHABET_DATASET_FOLDER))):
        print("input dataset already downloaded")
        return

    __download_data()
    __unzip_dataset()
    #__cleanup_data()
