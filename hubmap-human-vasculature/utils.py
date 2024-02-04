import os
import glob
import re
from PIL import Image

def clean_dataset(dir):
    '''
    Cleans the dataset folder by renaming numbered images to be in numerical order for easier Pytorch dataset definition
    '''
    # for file_num, file in enumerate(os.listdir('./data/train')):
    #     print(f'{file_num} {file}')

    for file in os.listdir(dir):
        if file[-4:] == ".jpg":
            image = Image.open(os.path.join(dir, file))
            image.save(os.path.join(dir, file[:-4] + ".png"), "PNG")
            os.remove(os.path.join(dir, file))
    
    files = glob.glob1(dir, "*.png")
    files = sorted(files, key=lambda x:float(re.findall("(\d+)",x)[0]))
    print(files)
    
    for file_num, file in enumerate(files):
        path = os.path.join(dir, file)
        os.rename(path, os.path.join(dir, str(file_num) + file[file.index('_'):]))
    
    print(os.listdir(dir))

        

clean_dataset("./test2")