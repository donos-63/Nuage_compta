import os
import zipfile
import requests
from PIL import Image
import numpy as np
import sys
import csv

alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']

# Définition des variables
DATAS_LOCAL_PATH = './DATAS/'
RAW_LOCAL_PATH = DATAS_LOCAL_PATH + 'RAW/'
ZIP_LOCAL_PATH = RAW_LOCAL_PATH + 'alphabet-dataset.zip'
MNIST_LOCAL_PATH = RAW_LOCAL_PATH + 'mnist.npz'
CURATED_LOCAL_PATH = DATAS_LOCAL_PATH + 'CURATED/'
DATASET_PATH = CURATED_LOCAL_PATH + 'dataset.csv'
MODELS_LOCAL_PATH = './MODELS/'
URL = 'https://stdatalake006.blob.core.windows.net/public/alphabet-dataset.zip'
URL_MNIST = 'https://raw.githubusercontent.com/thdiaman/deeplearning/master/OCR/mnist.npz'



def check_folder ():
    PATH = [DATAS_LOCAL_PATH, RAW_LOCAL_PATH, CURATED_LOCAL_PATH, MODELS_LOCAL_PATH]
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
        print('Letters datas already downloaded.')

    if os.path.exists(MNIST_LOCAL_PATH) == False:
        dl_data_mnist()
    else : 
        print('Numbers datas already downloaded.')
    
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

def dl_data_mnist ():
    print ('Downloading...')
    with open(MNIST_LOCAL_PATH, "wb") as f:
        r = requests.get(URL_MNIST)
        f.write(r.content)
    print ('Dataset dowloaded successfully.')

def extract_data(path):
    print (f'Extracting {path}...')
    with zipfile.ZipFile(ZIP_LOCAL_PATH, 'r') as z:
        z.extract(path, RAW_LOCAL_PATH)   
        for filename in z.namelist():
            if filename.startswith(path):
                z.extract(filename, RAW_LOCAL_PATH)

    print ('Successfull.')


def png_to_csv (car_path, number) :

    fullRawDir = []

    for cp in car_path :
        frd = f'{RAW_LOCAL_PATH}{cp}'
        fullRawDir.append(frd)

    form='.png'
    fileList = []

    for directory in fullRawDir :
        for root, dirs, files in os.walk(directory, topdown=False):
            n = 0
            for name in files:
                if n == number :
                    break
                else :
                    if name.endswith(form):
                        fullName = f'{root}{name}'
                        fileList.append(fullName)
                        n += 1
    
    if os.path.exists(DATASET_PATH) :
        os.remove(DATASET_PATH)

    with open(DATASET_PATH, 'a') as f:
        for filename in fileList:

            lettre = filename[29]
            label = alphabet.index(lettre)

            img_file = Image.open(filename)

            value = np.asarray(img_file.getdata(),dtype=np.int).reshape((img_file.size[1],img_file.size[0]))
            value = np.insert(value, 0, label)
            value = value.flatten()

            with open(DATASET_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(value)
    
    print ('All png files convert to a csv file.')