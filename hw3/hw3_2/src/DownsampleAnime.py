from imageio import imread, imsave
from skimage.transform import resize, rescale
import os

AnimePath = '../AnimeDataset'
original = os.listdir(os.path.join(AnimePath, 'faces'))

for image in original:
    print ('Resizing', image)
    path = os.path.join(AnimePath, 'faces', image)
    img = imread(path)
    img = rescale(img, 64.0/96.0, anti_aliasing=False)
    imsave(os.path.join(AnimePath, 'faces64', image), img)
