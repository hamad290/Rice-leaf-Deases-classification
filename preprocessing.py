
from PIL import Image
import numpy as np
from skimage.morphology import binary_closing, binary_opening, erosion
import glob
files = glob.glob('D://3set/dataset/*/*')
#files_reshape = list(map(lambda x: x.replace('/dataset\\', '/Resized\\'), files))
#basewidth = 300
#for file, file_save in zip(files, files_reshape):
    #img = Image.open(file)
    #wpercent = (basewidth/float(img.size[0]))
    #hsize = int((float(img.size[1])*float(wpercent)))
    #img = img.resize((basewidth,hsize))
    #img.save(file_save) 

from skimage.morphology import binary_closing, binary_opening, erosion
files_bgremoved = list(map(lambda x: x.replace('/dataset\\', '/backgroundremove\\'), files))
selem = np.zeros((25, 25))

ci,cj=12, 12
cr=13

## Create index arrays to z
I,J=np.meshgrid(np.arange(selem.shape[0]),np.arange(selem.shape[1]))

# #calculate distance of all points to centre
dist=np.sqrt((I-ci)**2+(J-cj)**2)

## Assign value of 1 to those points where dist<cr:
selem[np.where(dist<=cr)]=1

import numpy as np
from scipy import ndimage

# fig, ax = plt.subplots(20,2, figsize=(10,80))
idx = 0
for file, file_save in zip(files, files_bgremoved):
    bg_frac = 0
    thres = 220
    img = Image.open(file)
    im_arr = np.array(img)
#     ax[idx, 0].imshow(im_arr)
    R = im_arr[:, :, 0]
    G = im_arr[:, :, 1]
    B = im_arr[:, :, 2]
    while bg_frac < 0.6: 
        bg_mask = ((R>thres) | (B>thres))# & (G < 100)
        bg_frac = bg_mask.sum()/len(bg_mask.flatten())
        thres -= 5
    # we use opening first since our mask is reversed (the foreground and background are reversed here)
    bg_mask = binary_closing(erosion(binary_opening(bg_mask, selem), np.ones((3, 3))), np.ones((5,5)))
    
    #Get biggest blob
    label, num_label = ndimage.label(~bg_mask)
    size = np.bincount(label.ravel())
    biggest_label = size[1:].argmax() + 1
    bg_mask = label == biggest_label
    
    im_arr[~bg_mask, 0] = 255
    im_arr[~bg_mask, 1] = 255
    im_arr[~bg_mask, 2] = 255
    
    img = Image.fromarray(im_arr)
    img.save(file_save)
    idx+=1