from glob import glob
from itertools import permutations
from re import L
import os
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
import utils
import numpy as np


permutation = [''.join(p) for p in permutations('0123')]
file_path = '/Users/aniruddhasrinivasan/Downloads/Puzzles/shuffle/'
file_name = ''

for p in permutation:

    isExist = os.path.exists(file_path+p+'full')
    if (isExist==False):
         os.makedirs(file_path+p+'full')

dirlist = [ item for item in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, item)) ]
i = 0
for p in permutation:
    for dir in dirlist:
        files = [f for f in glob(file_path+dir+'/*')]
        random_files = np.random.choice(files, int(len(files)*.1))
        i=0
        for img_name in random_files:
            example_image = Image.open(img_name)
            if p == dir:
                example_image.save('shuffle/'+p+'full/'+dir+'_'+p+'_'+str(i)+'.png')
                continue
            pieces = utils.get_uniform_rectangular_split(np.asarray(example_image), 2, 2)
            idx0 = dir.find('0')
            idx1 = dir.find('1')
            idx2 = dir.find('2')
            idx3 = dir.find('3')
            idx_array = [dir.find('0'),dir.find('1'),dir.find('2'),dir.find('3')]
            final_image = Image.fromarray(np.vstack((np.hstack((pieces[idx_array[int(p[0])]],pieces[idx_array[int(p[1])]])),
            np.hstack((pieces[idx_array[int(p[2])]],pieces[idx_array[int(p[3])]])))))
            print(file_path+p+'_full/'+dir+'_'+p+'_'+str(i)+'.png')
            final_image.save('shuffle/'+p+'full/'+dir+'_'+p+'_'+str(i)+'.png')
            i+=1
