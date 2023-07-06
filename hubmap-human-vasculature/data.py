import json
import typing as t
import numpy as np
from PIL import Image

def load_masks(path: str) -> t.List:
    '''
    Loads in coordinates for segmentation masks for each image from the 'polygons.jsonl' file
    '''
    mask_coordinates = []

    with open(path) as f:
        for line in f:
            mask = json.loads(line)
            mask_coordinates.append(mask)
        
    return mask_coordinates

def find_categories(masks: t.List) -> t.List:
    '''
    Another helper function to iterate through the dataset and find out how many unique microvasculature instances there are
    '''
    instance_types = []

    for mask in masks:
        for instance in mask['annotations']:
            if instance['type'] not in instance_types:
                instance_types.append(instance['type'])
    
    return instance_types

def plot_masks(mask: t.Dict, id: str) -> np.ndarray:
    '''
    The coordinates of the instances in the masks are stored as long arrays.
    This is just a visualization function that turns them into images for easy comparison with model predictions.
    '''
    
    # Since there are three possible instances, the shape of each image mask will be (512, 512, 3)
    image = np.zeros((512, 512, 3), dtype=np.float32)

    encoded = {
        'glomerulus': [0.0, 0.0, 1.0],
        'blood_vessel': [0.0, 1.0, 0.0],
        'unsure': [1.0, 0.0, 0.0],
    }

    for annotation in mask['annotations']:
        coors = np.asarray(annotation['coordinates'])
        coors = np.squeeze(coors)

        for coor_pair in coors:
            image[coor_pair[0]][coor_pair[1]] = encoded[annotation['type']]
        
    
    mask_img = Image.fromarray((image * 255.0), 'RGB')
    mask_img.show()

    return image

result = load_masks('./data/polygons.jsonl')
# print(find_categories(result))
plot_masks(result[0], result[0]['id'])