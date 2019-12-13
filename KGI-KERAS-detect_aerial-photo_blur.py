# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import paths
import os
import gc
import argparse
import cv2
import numpy as np
from osgeo import gdal
import easygui
import fnmatch
import time
import sys

def color2gray_laplacian_windows(imagePath,x,y):
    gdal.Translate('/vsimem/clip.tif', imagePath, srcWin = [x, y, 500, 500])
    image = gdal.Open('/vsimem/clip.tif')
    a_image = image.ReadAsArray()
    if a_image.dtype == np.uint16:
       a_image = ( a_image/256).astype('uint8')

    s_image = np.dstack((a_image[0],a_image[1],a_image[2]))
    s_image = cv2.cvtColor(s_image,cv2.COLOR_RGB2BGR)
    GRAY = cv2.cvtColor(s_image,cv2.COLOR_BGR2GRAY)
    fmt = cv2.Laplacian(GRAY, cv2.CV_64F).var()
    gdal.Unlink('/vsimem/clip.tif')
    vue = False
    if vue :
       text = os.path.basename(imagePath) + "  Blurry image " +str(i) + "/" + str(total_con)
       rtext= os.path.basename(imagePath) + "  threshold = " + str("%.3f"%fmt) + "\n"
       cv2.rectangle(s_image, (10, 10), (490, 50), (200, 200, 200), -1)
       cv2.putText(s_image, "  {}  || {:.2f}".format(text, fmt), (10, 30),
         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)
       cv2.namedWindow('image',cv2.WINDOW_NORMAL)
       cv2.resizeWindow('image', 500,500)
       cv2.moveWindow('image', 40,30)
       cv2.imshow('image', s_image)
       key = cv2.waitKey(0)
       cv2.destroyAllWindows()
    return fmt

def spiral(N,M):
    x,y = 0,0   
    dx, dy = 0, -1

    for dumb in range(N*M):

        if abs(x) == abs(y) and [dx,dy] != [1,0] or x>0 and y == 1-x:  
            dx, dy = -dy, dx            # corner, change direction

        yield x, y
        x, y = x+dx, y+dy

def gdalK(imagePath,x,y,pixS,inter):
    gdal.Translate('/vsimem/clip.tif', imagePath, srcWin = [x, y, pixS, pixS])
    image = gdal.Open('/vsimem/clip.tif')
    a_image = image.ReadAsArray()
    if a_image.dtype == np.uint16:
       a_image = ( a_image/256).astype('uint8')
    
    image = np.dstack((a_image[0],a_image[1],a_image[2]))
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (inter, inter))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    gdal.Unlink('/vsimem/clip.tif')
    return image

def urbanNoturban(imagePath,XW,YH):

    # 01 center_0/0
    x =  (XW/2 - 600)
    y =  (YH/2 - 600)
    xl = (XW/2 - 250)
    yl = (YH/2 - 250)
    fmBest = 0
    isUrbam = False
    UrbamNum = 0
    #print('center :'+str(xl)+' / '+str(yl))

    for i,ii in spiral(11,11):
    #for i in range(-5, 6):
     #for ii in range(-5,6):
       nx = x+(i*1200)
       ny = y+(ii*1200)
       nxl = nx-(350)
       nyl = ny-(350)
       #print(str(i[0])+' / '+str(i[1]))
       #print(str(nxl)+' / '+str(nyl))
     
       image =  gdalK(imagePath,nx,ny,1200,300)
       (notUrban, urban) = model.predict(image)[0]
       #print('notUrban :'+str(notUrban)+' / '+'urban :'+str(urban))
        
       if urban > 0.9 :
         isUrbam = True
         UrbamNum += 1
         fm = color2gray_laplacian_windows(imagePath,nxl,nyl)
         fmBest = fmBest+fm
         #if fm < fmBest : fmBest = fm
  

    if isUrbam :   
       return fmBest/UrbamNum
    else :
       return 100


def cls():
    os.system('cls' if os.name=='nt' else 'clear')




def update_progress(progress):
    barLength = 30 # Modify this to change the length of the progress bar
    
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% ".format( "#"*block + "-"*(barLength-block), int(progress*100))
    sys.stdout.write(text)
    sys.stdout.flush()


dirname = easygui.diropenbox(msg=None, title="Please select a directory", default=None)
total_con=len(fnmatch.filter(os.listdir(dirname), '*.tif'))
msg = str(total_con) +" files do you want to continue?"
title = "Please Confirm"
if easygui.ynbox(msg, title, ('Yes', 'No')): # show a Continue/Cancel dialog
    pass # user chose Continue else: # user chose Cancel
else:
    exit(0)

file_Dir = os.path.basename(dirname)
f = open(dirname+"/"+file_Dir+"-result-blur.txt", "w")
i=0
model = load_model('kgi-urban-rural.model')
cls()
# loop over the input images


for imagePath in paths.list_images(dirname):
 i = i+1
 dataset = gdal.Open(imagePath, gdal.GA_ReadOnly)
 xwidth=(dataset.RasterXSize)
 yheight=(dataset.RasterYSize)
 dataset = None

 
 fm = urbanNoturban(imagePath,xwidth,yheight)

 if fm < 10:
       text= os.path.splitext(os.path.basename(imagePath))[0] + "    " + str("%.3f"%fm) + "\n"
       f.write(text)


 update_progress(i/total_con)
f.close()
print('Done')

