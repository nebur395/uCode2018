import argparse
import PIL
from PIL import Image
import numpy as np
import os.path
import glob
import random
import ntpath
import cv2




for filename in glob.glob( 'labels/*/*.png'):
	print(filename)

	img = cv2.imread(filename)
	img[img == 1]=255
	img[img == 5]=1
	img[img == 6]=255
	img[img == 7]=5
	img[img == 8]=255
	img[img == 9]=255
	img[img == 10]=255
	img[img == 11]=4
	img[img == 12]=5
	img[img == 13]=4
	img[img == 14]=1
	img[img == 15]=255
	img[img == 16]=5
	img[img == 17]=255
	img[img == 18]=255
	img[img == 19]=6
	img[img == 20]=7
	img[img == 21]=5
	img[img == 22]=8
	img[img == 23]=255
	img[img == 24]=4
	img[img == 25]=3
	img[img == 26]=8
	img[img == 27]=9
	img[img == 28]=5
	img[img == 29]=255
	img[img == 30]=255
	img[img == 31]=3
	img[img == 32]=5
	img[img == 33]=255
	img[img == 34]=255
	img[img == 35]=1
	img[img == 36]=5
	img[img == 37]=2
	img[img == 38]=8
	img[img == 39]=5
	img[img == 40]=3
	img[img == 41]=9
	img[img == 42]=10
	img[img == 43]=5
	img[img == 44]=255
	img[img == 45]=9
	img[img == 46]=255
	img[img == 47]=255
	img[img == 48]=8
	img[img == 49]=8
	img[img == 50]=255
	img[img == 51]=8
	img[img == 52]=255
	img[img == 53]=9
	img[img == 54]=8
	img[img == 55]=8
	img[img == 56]=255
	img[img == 57]=255
	img[img == 58]=5
	cv2.imwrite(filename,img)

