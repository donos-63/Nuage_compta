import os
import zipfile
import requests



# Définition des variables
DATAS_LOCAL_PATH = './DATAS/'
RAW_LOCAL_PATH = DATAS_LOCAL_PATH + 'RAW/'
ZIP_LOCAL_PATH = RAW_LOCAL_PATH + 'alphabet-dataset.zip'
CURATED_LOCAL_PATH = DATAS_LOCAL_PATH + 'CURATED/'
URL = 'https://stdatalake006.blob.core.windows.net/public/alphabet-dataset.zip'



def check_folder ():
    PATH = [DATAS_LOCAL_PATH, RAW_LOCAL_PATH, CURATED_LOCAL_PATH]
    for p in PATH:
        if not os.path.exists(p):
            os.mkdir(p)


def ensure_data_loaded(car_path):
    '''
    Ensure if data are already loaded. Download if missing
    '''
    if os.path.exists(ZIP_LOCAL_PATH) == False:
        dl_data()
    else :
        print('Datas already downloaded.')
    
    for cp in car_path :
        if os.path.exists(f'{RAW_LOCAL_PATH}{cp}') == False:
            extract_data(cp)

    if os.path.exists(f'{RAW_LOCAL_PATH}alphabet-dataset/handwritten-data.csv') == False:
        extract_data('alphabet-dataset/handwritten-data.csv')

    print ('Datas are successfully loaded.\n')


def dl_data ():
    print ('Downloading...')
    with open(ZIP_LOCAL_PATH, "wb") as f:
        r = requests.get(URL)
        f.write(r.content)
    print ('Dataset dowloaded successfully.')


def extract_data(path):
    print (f'Extracting {path}...')
    with zipfile.ZipFile(ZIP_LOCAL_PATH, 'r') as z:
        z.extract(filename, RAW_LOCAL_PATH)   
        for filename in z.namelist():
            if filename.startswith(path):
                z.extract(filename, RAW_LOCAL_PATH)

    print ('Successfull.')


