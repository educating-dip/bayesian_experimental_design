"""
Provides the RectanglesDataset.
"""
import numpy as np
import torch
from sklearn.feature_extraction import image
from itertools import repeat
from .walnuts_interface import get_ground_truth

class WalnutPatchesDataset(torch.utils.data.IterableDataset):
    """
    Dataset with images of multiple random rectangles.
    The images are normalized to have a value range of ``[0., 1.]`` with a
    background value of ``0.``.
    """
    def __init__(self, data_path='./', image_size=128, max_patches=32, walnut_id=1, orbit_id=2, slice_ind=253, fixed_seed=1):
        super().__init__()
        self.shape = (image_size, image_size)
        self.max_patches = max_patches
        self.walnut = get_ground_truth(data_path, walnut_id, orbit_id, slice_ind)[72:424, 72:424]
        self.fixed_seed = fixed_seed
        
    def __iter__(self):
        # note: this dataset will return the same images for each worker if using multiprocessing and a fixed seed
        r = np.random.RandomState(self.fixed_seed)
        it = repeat(None, self.max_patches) if self.max_patches is not None else repeat(None)
        for _ in it:
            patch = image.extract_patches_2d(
                self.walnut, 
                self.shape,
                max_patches=1,
                random_state=r)
            yield patch
