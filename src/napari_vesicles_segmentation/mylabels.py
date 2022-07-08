# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 17:46:43 2022

@author: andrea
"""

from _widget import Segmentation
import napari
from napari.layers import Image,Labels, Shapes
from magicgui import magicgui
import numpy as np
import tifffile as tiff
import pathlib
import os

@magicgui(call_button='Extract',)
def extract_rois(labels_layer:Labels,
                 folder=pathlib.Path(os.getcwd()),
                 sizez=10,
                 sizexy =30):
    from skimage.measure import regionprops_table
    if labels_layer.ndim==4:
        labels_stack = np.squeeze(labels_layer.data)
    else:
        labels_stack = labels_layer.data
    properties = ['centroid','area']

    centroids = regionprops_table(labels_stack, properties=properties)
    zs = centroids['centroid-0']
    ys = centroids['centroid-1']
    xs = centroids['centroid-2']
    areas = centroids['area']
    name = labels_layer.name.replace('_vesicles','')
    
    stack = viewer.layers[name].data
    index = 0
    for z,y,x,area in zip(zs,ys,xs,areas):
        roi = stack[int(z-sizez//2):int(z+sizez//2),
                    int(y-sizexy//2):int(y+sizexy//2),
                    int(x-sizexy//2):int(x+sizexy//2),
                    ]
        if not os.path.isdir(folder):
            os.makedirs(folder)
        path = os.path.join(folder, f'{name}_{index}.tif') 
        index +=1
        tiff.imwrite(path, roi)
    print(f'Saved {index} ROIs')
    

# Creates a viewer
viewer = napari.Viewer()
# Adds segmentation widget to the viewer
segmentator = Segmentation(viewer)
viewer.window.add_dock_widget(segmentator, name="Vesicles segmentation", add_vertical_stretch=True)
viewer.window.add_dock_widget(extract_rois, name="Extract ROIs", add_vertical_stretch=True)










# Run napari
napari.run()