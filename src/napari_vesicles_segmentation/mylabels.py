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
                 source_folder=pathlib.Path(os.getcwd()),
                 destination_folder=pathlib.Path(os.getcwd()),
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
    sim_name = labels_layer.name.replace('_vesicles','')
    wf_name = sim_name.replace('_sim','_wf')
    
    
    stack = viewer.layers[sim_name].data
    index = 0
    
    wf_path = os.path.join(source_folder, f'{wf_name}.tif') 
    wf_stack = tiff.imread(wf_path)
    
    
    for z,y,x,area in zip(zs,ys,xs,areas):
        sim_roi = stack[int(z-sizez//2):int(z+sizez//2),
                    int(y-sizexy//2):int(y+sizexy//2),
                    int(x-sizexy//2):int(x+sizexy//2),
                    ]
        wf_roi = wf_stack[int(z-sizez//2):int(z+sizez//2),
                    int(y-sizexy//2):int(y+sizexy//2),
                    int(x-sizexy//2):int(x+sizexy//2),
                    ]
        if not os.path.isdir(destination_folder):
            os.makedirs(destination_folder)
        destination_sim_path = os.path.join(destination_folder, f'{sim_name}_{index}.tif') 
        destination_wf_path = os.path.join(destination_folder, f'{wf_name}_{index}.tif') 
        
        index +=1
        tiff.imwrite(destination_sim_path, sim_roi)
        tiff.imwrite(destination_wf_path, wf_roi)
    print(f'Saved {index} ROIs')
    

# Creates a viewer
viewer = napari.Viewer()
# Adds segmentation widget to the viewer
segmentator = Segmentation(viewer)
viewer.window.add_dock_widget(segmentator, name="Vesicles segmentation", add_vertical_stretch=True)
viewer.window.add_dock_widget(extract_rois, name="Extract ROIs", add_vertical_stretch=True)










# Run napari
napari.run()