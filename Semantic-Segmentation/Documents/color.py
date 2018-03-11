import argparse
import PIL
from PIL import Image
import numpy as np
import os.path
import glob
import random
import ntpath
import cv2
import copy

'''
1 vestido
2 bolso
3 pantalon
4 chaqueta
5 zapas
6 pelo
7 sobrero
8 camiseta
9 legis o piernas o brazos
10 falda
'''
etiqueta = 8

for filename in glob.glob( './labels/IMG_20180309_214234.*'):
	print(filename)
	label = cv2.imread(filename)
	img = cv2.imread(filename.replace('labels','images').replace('.jpg','.jpg'))
	img_filt = cv2.imread(filename.replace('labels','images').replace('.jpg','_lad.jpg'))
	print(filename.replace('labels','images').replace('.png','nick.png'))
	print(img_filt)
	mask = copy.deepcopy(label)
	mask_not = copy.deepcopy(label)
	mask[label != etiqueta]=0
	mask[label == etiqueta]=1
	mask_not[label == etiqueta]=0
	mask_not[label != etiqueta]=1
	print(np.unique(mask_not))
	prenda_enmascarada = img*mask # pasar esta imagen a el metodo
    	cv2.imwrite('ropa.jpg',prenda_enmascarada)
	# hacer el filtro con la prenda enmascarda
	prenda_enmascarada = img_filt*mask # pasar esta imagen a el metodo
    	cv2.imwrite('ropa_filtro.jpg',prenda_enmascarada)
	img=img*mask_not

	img = img + prenda_enmascarada
    	cv2.imwrite('total.jpg',img)
	cv2.imshow('image', prenda_enmascarada )
	cv2.waitKey(0)
	cv2.destroyAllWindows()


