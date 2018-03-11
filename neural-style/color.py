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
etiqueta = 5

for filename in glob.glob( 'person*.jpg'):
	label = cv2.imread(filename,0)
	label[label<10]=8
	label[label>=10]=0
	print(label.shape)
	cv2.imshow('image', label )
	cv2.waitKey(0)
	cv2.destroyAllWindows()
    	cv2.imwrite(filename,label)



