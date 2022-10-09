import numpy as np
import tensorflow as tf
from glob import glob
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
from itertools import permutations


def get_pieces(img, rows, cols, row_cut_size, col_cut_size):
    pieces = []
    for r in range(0, rows, row_cut_size):
        for c in range(0, cols, col_cut_size):
            pieces.append(img[r:r+row_cut_size, c:c+col_cut_size, :])
    return pieces

# Splits an image into uniformly sized puzzle pieces
def get_uniform_rectangular_split(img, puzzle_dim_x=2, puzzle_dim_y=2):
    rows = img.shape[0]
    cols = img.shape[1]
    if rows % puzzle_dim_y != 0 or cols % puzzle_dim_x != 0:
        print('Please ensure image dimensions are divisible by desired puzzle dimensions.')
    row_cut_size = rows // puzzle_dim_y
    col_cut_size = cols // puzzle_dim_x

    pieces = get_pieces(img, rows, cols, row_cut_size, col_cut_size)

    return pieces

##loading images from folders into variables, as (4,64,64,3) images
def load_images_from(path,label,maxsample=np.inf):
    i=0
    x=[]
    for img_name in glob(path+label+'/*'):
            img=Image.open(img_name)
            x.append(np.stack(get_uniform_rectangular_split(img_to_array(img))))
            #img=load_img(path+label+'/'+img_name)
            i+=1
            if i>maxsample:
                break
    y=[[int(i) for i in label]]*len(x)
    return (np.array(x), np.expand_dims(np.array(y), axis=-1))


def load_all(path,maxsample=np.inf):
    X=[]
    Y=[]
    for label in [''.join(str(x) for x in comb) for comb in list(permutations(range(0, 4)))]:
        x,y=load_images_from(path,label,maxsample)
        X.append(x)
        Y.append(y)
    return (np.vstack(X),np.vstack(Y))