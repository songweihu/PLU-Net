import random
import shutil
import os
from glob import glob
import cv2 as cv

def det_seg(PATH, sort, test, train_rate):
    '''

    :param: PATH:str(the image path)
    :param: sort:str(the dataset type)
    :param: test:bool(if need test dataset or not)
    :param: train_rate:int(the rate of train dataset)
    '''
    dirs = os.listdir(PATH)
    images = PATH + "images"
    mask = PATH + "/masks"
    
    files = glob(images+"/*")
    masks = glob(mask+"/*")
    
    TRAIN_PATH = PATH + "/train"
    VALID_PATH = PATH + "/valid"
    TEST_PATH = PATH + "/test"

    TRAIN_IMAGES = TRAIN_PATH + "/images/"
    TRAIN_MASKS = TRAIN_PATH + "/masks/"
    VALID_IMAGES = VALID_PATH+ "/images/"
    VALID_MASKS = VALID_PATH+ "/masks/"
    TEST_IMAGES = TEST_PATH+ "/images/"
    TEST_MASKS = TEST_PATH+ "/masks/"
    
    os.makedirs(TRAIN_PATH, exist_ok=True) 
    os.makedirs(VALID_PATH, exist_ok=True)
    os.makedirs(TRAIN_IMAGES, exist_ok=True)
    os.makedirs(TRAIN_MASKS, exist_ok=True)
    os.makedirs(VALID_IMAGES, exist_ok=True)
    os.makedirs(VALID_MASKS, exist_ok=True)

    random.seed(42)
    random.shuffle(files)
    random.shuffle(files)
    train_len = int(len(files)*train_rate)
    if test:
        os.makedirs(TEST_PATH, exist_ok=True)
        valid_len = ((1 - train_rate) / 2 + train_rate) * len(files)
        os.makedirs(TEST_IMAGES, exist_ok=True)
        os.makedirs(TEST_MASKS, exist_ok=True)
    else:
        valid_len = len(files)

    for idx, path in enumerate(files):
        print(len(files), train_len, valid_len) 
        if idx <= train_len:
            shutil.copy(path, TRAIN_IMAGES)
            mask_path = "skin_dataset/masks/"+os.path.basename(path).split(".")[0]+".png"
            shutil.copy(mask_path, TRAIN_MASKS)
        elif train_len <= idx and idx <= valid_len:
            shutil.copy(path, VALID_IMAGES)
            mask_path = "skin_dataset/masks/"+os.path.basename(path).split(".")[0]+".png"
            shutil.copy(mask_path, VALID_MASKS)
        else:
            shutil.copy(path, TEST_IMAGES)
            mask_path = "skin_dataset/masks/"+os.path.basename(path).split(".")[0]+".png"
            shutil.copy(mask_path, TEST_MASKS)

    '''
    for idx, path in enumerate(masks):
        if idx <= train_len:
            shutil.copy(path, TRAIN_MASKS)
        elif train_len <= idx and idx <= valid_len:
            shutil.copy(path, VALID_MASKS)
        else:
            shutil.copy(path, TEST_MASKS)
    '''
def cls(PATH, sort, test, train_rate):
    '''

    :param: PATH:str(the image path)
    :param: sort:str(the dataset type)
    :param: test:bool(if need test dataset or not)
    :param: train_rate:int(the rate of train dataset)
    '''
    dirs = os.listdir(PATH)
    for dir_name in dirs:
        images = PATH + dir_name
        files = glob(images+"/*")

        TRAIN_PATH = PATH + "/train/"
        VALID_PATH = PATH + "/valid/"
        TEST_PATH = PATH + "/test/"

        TRAIN_IMAGES = TRAIN_PATH + dir_name 
        VALID_IMAGES = VALID_PATH+ dir_name
        TEST_IMAGES = TEST_PATH+ dir_name
    
        os.makedirs(TRAIN_PATH, exist_ok=True) 
        os.makedirs(VALID_PATH, exist_ok=True)
        os.makedirs(TRAIN_IMAGES, exist_ok=True)
        os.makedirs(VALID_IMAGES, exist_ok=True)

        random.seed(42)
        random.shuffle(files)
        random.shuffle(files)
        train_len = int(len(files)*train_rate)
        if test:
            os.makedirs(TEST_PATH, exist_ok=True)
            valid_len = ((1 - train_rate) / 2 + train_rate) * len(files)
            os.makedirs(TEST_IMAGES, exist_ok=True)
        else:
            valid_len = len(files)

        for idx, path in enumerate(files):
            print(len(files), train_len, valid_len) 
            if idx <= train_len:
                shutil.copy(path, TRAIN_IMAGES)
            elif train_len <= idx and idx <= valid_len:
                shutil.copy(path, VALID_IMAGES)
            else:
                shutil.copy(path, TEST_IMAGES)

def split_dataset(PATH, sort="det", test=False, train_rate=0.8):
    '''

    :param: PATH:str(the image path)
    :param: sort:str(the dataset type)
    :param: test:bool(if need test dataset or not)
    :param: train_rate:int(the rate of train dataset)
    '''
    if sort == "cls":
        cls(PATH, sort, test, train_rate)
    else:
        det_seg(PATH, sort, test, train_rate)

if __name__ == "__main__":
    PATH = "skin_dataset/"
    split_dataset(PATH, "det", True, 0.6)
