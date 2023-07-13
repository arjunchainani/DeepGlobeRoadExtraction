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

def plot_segmented_masks(mask: t.Dict):
    '''
    def get_cartesian_coords(coords, img_height):
    coords_array = np.array(coords).squeeze()
    xs = coords_array[:, 0]
    ys = -coords_array[:, 1] + img_height
    
    return xs, ys

def plot_annotated_image(image_dict, scale_factor: int = 1.0) -> None:
    #array = tiff.imread(CFG.img_path_template.format(image_dict["id"]))
    array = tiff.imread(f'/kaggle/input/hubmap-hacking-the-human-vasculature/train/{image_dict["id"]}.tif')
    
    img_example = Image.fromarray(array)
    annotations = image_dict["annotations"]
    
    # create figure
    fig = go.Figure()

    # constants
    img_width = img_example.size[0]
    img_height = img_example.size[1]
    

    # add invisible scatter trace
    fig.add_trace(
        go.Scatter(
            x=[0, img_width],
            y=[0, img_height],
            mode="markers",
            marker_opacity=0
        )
    )

    # configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, img_height],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # add image
    fig.add_layout_image(dict(
        x=0,
        sizex=img_width,
        y=img_height,
        sizey=img_height,
        xref="x", yref="y",
        opacity=1.0,
        layer="below",
        sizing="stretch",
        source=img_example
    ))
    
    # add polygons
    for annotation in annotations:
        name = annotation["type"]
        xs, ys = get_cartesian_coords(annotation["coordinates"], img_height)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill="toself",
            name=name,
            hovertemplate="%{name}",
            mode='lines'
        ))

    # configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False
    )

    # disable the autosize on double click because it adds unwanted margins around the image
    # and finally show figure
    fig.show(config={'doubleClick': 'reset'})
plot_annotated_image(tiles_dicts[0])
make s
    '''
    
    pass

def plot_annotations(mask: t.Dict, id: str, stack_image: bool = False) -> np.ndarray:
    '''
    The coordinates of the instances in the masks are stored as long arrays.
    This is just a visualization function that turns them into images for easy comparison with model predictions.

    Takes in a parameter called stack_image which determines whether to display the masks by themselves or pasted on top of the original image
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

    if stack_image:
        original_img = Image.open(f'./data/train/{mask["id"]}.tif')
        
        # To properly paste the mask, the unused pixels in the mask image need to be transparent
        # To do this, we convert the image to an RGBA color space and then turn the opacity of all black pixels to 0
        mask_img = mask_img.convert('RGBA')
        mask_data = mask_img.getdata()
        transparent = []

        for data in mask_data:
            if data[0] == 0 and data[1] == 0 and data[2] == 0:
                transparent.append((0, 0, 0, 0))
            else:
                # Making sure any actual annotation pixels have the maximum alpha value
                transparent.append((data[0], data[1], data[2], 255))
        
        mask_img.putdata(transparent)

        # Now we can paste the transparent mask onto the original image
        original_img.paste(mask_img, (0, 0), mask=mask_img)   
        original_img.show()

    else: 
        mask_img.show()

    return image

if __name__ == '__main__':
    result = load_masks('./data/polygons.jsonl')
    plot_masks(result[500], result[500]['id'], stack_image=True)