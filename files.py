import os

def check_files():
    '''
    Checks whether all files from the dataset have been successfully installed.
    Needs to be run from a terminal with administrator access
    '''
    downloaded = os.listdir('./data/train')
    names = []
    # kaggle_download = os.listdir('./hubmap-hacking-the-human-vasculature (1)/train')

    # for file in kaggle_download:
        # if file not in downloaded:
            # print(file)

    for file in downloaded:
        if file in names:
            print(file)
        else:
            names.append(file)

    print(downloaded)

check_files()