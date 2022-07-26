import napari
from qtpy.QtWidgets import QWidget, QVBoxLayout
from napari.layers import Image
from napari.qt.threading import thread_worker
from magicgui import magicgui
from skimage import filters, transform
import skimage.morphology as morph
from skimage import measure
import scipy.ndimage as ndi
import numpy as np
from time import perf_counter


def detect_cell(im: np.ndarray, membrane_erosion: int, closing_size: int, n_sigma: int, downsizing_ratio: int) -> np.ndarray:
    """
    This function detects the cell in the image using the Otsu thresholding method. The external membrane is removed by closing, filling holes and eroding the detected cell. For faster computation, these operations can be performed in a downsampled image.
    :param im: The image to detect the cell in.
    :param membrane_erosion: The size of the disk radius used for eroding the cell. This is used to remove the external membrane. This parameter scales when downsizing the image, for more information see 'downsizing_ratio' parameter.
    :param closing_size: The size of the disk radius used for closing the cell. This is used to fill holes in the cell. This parameter scales when downsizing the image, for more information see 'downsizing_ratio' parameter.
    :param n_sigma: If set to zero, no standardization is performed. Otherwise, the standard deviation of the image is set to n_sigma * the standard deviation of the image, the image is standardized and its values are clipped to the range [-1, 1] in order to remove outliers. The higher the value of n_sigma, the less outliers are removed. This operation can lead to a better detection of the cell.
    :param downsizing_ratio: The downsampling ratio used for the downsampled image. This is used to speed up the computation. Downsampling the image have impact in reducing the resolution of erosion and closing e.g. for a downsize ratio of 2, setting the erosion size to 3 will result in an erosion size of 6.
    :return: A binary image representing the cell detected mask.
    """
    # Downsize the image for faster computation
    downsized_im = transform.resize(im, (im.shape[1] // downsizing_ratio, im.shape[0] // downsizing_ratio), anti_aliasing=False)

    # Standardize and clip the image values to remove extreme ones
    if n_sigma > 0:
        downsized_im = (downsized_im - np.mean(downsized_im)) / (np.std(downsized_im) * n_sigma)
        downsized_im = downsized_im.clip(min=-1, max=1)
    # Normalize the downsized_im between 0 and 1
    downsized_im = (downsized_im - downsized_im.min()) / (downsized_im.max() - downsized_im.min())

    # Detect the cell using Otsu thresholding and apply morphological operations in order to remove the external membrane
    cell = downsized_im > filters.threshold_otsu(downsized_im)
    if closing_size > 0:
        cell = morph.binary_closing(cell, morph.disk(closing_size))
    cell = ndi.binary_fill_holes(cell)
    if membrane_erosion > 0:
        cell = morph.binary_erosion(cell, morph.disk(membrane_erosion))

    # Upsize the image to original size and return it as a binary image
    return transform.resize(cell.astype(np.uint8), (im.shape[1], im.shape[0]), anti_aliasing=False) > 0


class Segmentation(QWidget):
    # Constructor
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        layout = QVBoxLayout()
        self.viewer.layers.events.inserted.connect(self.segment.reset_choices)
        self.viewer.layers.events.removed.connect(self.segment.reset_choices)
        layout.addWidget(self.segment.native)
        self.setLayout(layout)
        
    
    @magicgui(call_button='Run segmentation',
              min_size={'label': 'minimum vesicles size'},
              n_sigma={'label': 'clip[std]'},
              downsizing_ratio={'min': 1}) 
    def segment(self, image: Image,
                min_size: int= 0,
                membrane_erosion: int=3,
                closing_size: int=0,
                n_sigma: int=3,
                downsizing_ratio: int=4,
                display_cell_detection: bool=False):
        """
        This function segments vesicles in the image by detecting the cell and then detecting the vesicles.
        The cell is detected using the Otsu thresholding method. The external membrane is removed by closing, filling holes and eroding the detected cell. For faster computation, these operations can be performed in a downsampled image.
        The vesicles are detected by performing an Otsu thresholding on the detected cell.
        This function adds the detected vesicles or the detected cell (set in the display_cell_detection parameter) to the viewer.
        :param image: The image to segment vesicles in. The image can be a 2D or 3D temporal stack of images.
        :param min_size: The minimum size of the vesicles to detect. Smaller detected vesicles are removed.
        :param membrane_erosion: Amount of pixels to erode the cell. Changing downsizing scales this parameter.
        :param closing_size: Cell clozing disk radius. Changing downsizing scales this parameter.
        :param n_sigma: If zero, no clip is performed. Otherwise, the higher the value, the less outliers are removed.
        :param downsizing_ratio: This is used to speed up the computation. Downsampling the image have impact in reducing the resolution of erosion and closing e.g. for a downsize ratio of 2, setting the erosion size to 3 will result in an erosion size of 6.
        :param display_cell_detection: If set to True, display the detected cell instead of vesicles.
        """
        def _add_labels(labels: tuple):
            """
            This function adds the labels to the viewer. If the layer already exists, it is updated.It is used as a callback function for the thread_worker. The labels in the first element of the tuple are added to the viewer as a label layer with the given name in the second element of the tuple.
            :param labels: A tuple containing the labels and the name of the layer.
            """
            if labels[1] in self.viewer.layers:
                self.viewer.layers[labels[1]].data = labels[0]
            else:
                self.viewer.add_labels(labels[0], name=labels[1])

        @thread_worker(connect={"returned": _add_labels})
        def _segment():
            """
            This function performs the computation of the segmentation of the vesicles of the given image in a separate thread to avoid blocking the GUI.
            :return: A tuple containing the labels and the name of the label layer.
            """
            start = perf_counter()
            ############################################################################################################
            #### Preprocessing
            ##################
            # If the image is single, convert it to a stack with a single slice
            ...
            if len(image.data.shape) == 2:
                stack = np.expand_dims(image.data, axis=0)
            else:
                stack = image.data
            # Normalize the image data between 0 and 1
            stack = (stack - stack.min()) / (stack.max() - stack.min())
            

            ############################################################################################################
            #### Segmentation
            #################
            # Detect the cell and erode the membrane if needed
            max_intensity_stack = np.max(stack, axis=0)
            max_intensity_cell = detect_cell(max_intensity_stack, membrane_erosion, closing_size, downsizing_ratio=1, n_sigma=n_sigma)
            # Segmentation of the vesicles
            vesicles_threshold = filters.threshold_otsu(max_intensity_stack[max_intensity_cell])
            vesicles = stack > vesicles_threshold

            ############################################################################################################
            #### Postprocessing
            ###################
            # Cell membrane detection and erosion
            for index in range(len(vesicles)):
                cell = detect_cell(stack[index], membrane_erosion, closing_size, downsizing_ratio=downsizing_ratio, n_sigma=n_sigma)
                if display_cell_detection:
                    vesicles[index] = cell
                else:
                    vesicles[index] *= cell
            # Remove small vesicles and false positives
            if min_size > 0:
                for index in range(len(vesicles)):
                    vesicles[index] = morph.remove_small_objects(vesicles[index], min_size=min_size)

            ############################################################################################################
            #### Return the result
            ######################
            labels = measure.label(vesicles.astype(np.uint8).squeeze())
            labels_name = f"{image.name}_{'cell' if display_cell_detection else 'vesicle'}"
            print(f"Segmentation of {image.name} took {perf_counter() - start:.2f} seconds.")
            return labels, labels_name
        _segment()


if __name__ == "__main__":
    # Creates a viewer
    viewer = napari.Viewer()

    # Adds segmentation widget to the viewer
    segmentator = Segmentation(viewer)
    viewer.window.add_dock_widget(segmentator, name="Vesicles segmentation", add_vertical_stretch=True)

    # Run napari
    napari.run()