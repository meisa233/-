import pylab
import imageio


import skimage
import numpy as np

filename = '/data/sv2007sbtest1/BG_2408.mpg'

vid = imageio.get_reader(filename, 'ffmpeg')
index = 0
for num, im in enumerate(vid):

    #print im.mean()
    #image = skimage.img_as_float(im).astype(np.float64)
    #fig = pylab.figure()
    #fig.suptitle('image #{}'.format(num), fontsize=20)
    #pylab.imshow(im)
    index += 1
pylab.show()
