# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 20:53:17 2016
@author: George Oliveira Barros, george_gob@hotmail.com. 
PGCA - Universidade Estadual de Feira de Santana (UEFS)

PathoSpoter-K Feature Extraction: Version 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
import ia636 as ia
import pymorph as morph

from skimage.feature import blob_log
from pylab import uint8
from skimage.filter import threshold_otsu
from skimage import img_as_ubyte #Conversão
from PIL import Image
from skimage.morphology import disk
from skimage.color import rgb2hed
from skimage.measure import label, regionprops 

print __doc__

glom_rgb = Image.open('s3.jpg')
glom_gl = glom_rgb.convert('L') #Gray Level

glom_hed = rgb2hed(glom_rgb) #hed
glom_h =  glom_hed[:, :, 0] #hematoxylim
glom_h = ia.ianormalize(glom_h)
selem = disk(10) #elemento estruturante
glom_h = np.array(glom_h) 
glom_h = 255 - uint8(glom_h)

#Segmentation
glom_by_reconsTopHat = morph.closerecth(glom_h,selem) #reconstrução morfológicas de fechamento

global_thresh = threshold_otsu(glom_by_reconsTopHat) #Otsu
glom_bin = glom_by_reconsTopHat > global_thresh + global_thresh*0.1 
glom_bin = img_as_ubyte(glom_bin)
selem = disk(3)    
glom_seg = morph.open(glom_bin, selem)
glom_seg = morph.close(glom_seg, selem) #Fechamento final

#Mostra as etapas
fig, axes = plt.subplots(2, 3, figsize=(14, 10))
fig.suptitle('Preprocessing, segmentation') 
ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()
ax1.imshow(glom_rgb, vmin=0, vmax=255, cmap=plt.cm.gray); ax1.set_title("RGB")
ax2.imshow(glom_hed, vmin=0, vmax=255, cmap=plt.cm.gray); ax2.set_title("HED") 
ax3.imshow(glom_h,  cmap=plt.cm.gray); ax3.set_title("255 - H (Hematoxylin)") 
ax4.imshow(glom_by_reconsTopHat, vmin=0, vmax=255, cmap=plt.cm.gray); ax4.set_title("Closing-by-reconstruction top-hat") 
ax5.imshow(glom_bin, vmin=0, vmax=255, cmap=plt.cm.gray); ax5.set_title("Otsu Thresholding") 
ax6.imshow(glom_seg, vmin=0, vmax=255, cmap=plt.cm.gray); ax6.set_title("Opening") 
#------------------------------------------------------------------------------------------------
#Feature Extraction
print "Feature Extraction:"
label_img = label(glom_seg) #Identifica formas através da vizinhaça 4 E 8
regions = regionprops(label_img) #Lista de propriedades das regiões selecionadas
print "Quantidade de regioes de nucleos", len(regions)
regionBlob =  img_as_ubyte(glom_seg)
blobs = blob_log(regionBlob, 20, threshold=.01)
print "Quantidade de aglomerados de nucleos", len(blobs)

plt.show()



