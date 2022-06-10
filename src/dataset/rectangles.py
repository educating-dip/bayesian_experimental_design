"""
Provides the RectanglesDataset.
"""
import numpy as np
import torch
from itertools import repeat
from skimage.draw import polygon
from skimage.transform import downscale_local_mean

def rect_phantom(shape, rects, smooth_sr_fact=8, blend_mode='add'):
    sr_shape = (shape[0] * smooth_sr_fact, shape[1] * smooth_sr_fact)
    img = np.zeros(sr_shape, dtype='float32')
    for rect in rects:
        v, a1, a2, x, y, rot = rect
        # convert [-1., 1.]^2 coordinates to [0., sr_shape[0]] x [0., sr_shape[1]]
        x, y = 0.5 * sr_shape[0] * (x + 1.), 0.5 * sr_shape[1] * (y + 1.)
        a1, a2 = 0.5 * sr_shape[0] * a1, 0.5 * sr_shape[1] * a2
        # rotate side vector [a1, a2] to rot_mat @ [a1, a2]
        rot_mat = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        coord_diffs = np.array([rot_mat @ [a1, a2], rot_mat @ [a1, -a2], rot_mat @ [-a1, -a2], rot_mat @ [-a1, a2]])
        coords = np.array([x, y])[None, :] + coord_diffs
        p_rr, p_cc = polygon(coords[:, 1], coords[:, 0], shape=sr_shape)
        if blend_mode == 'add':
            img[p_rr, p_cc] += v
        elif blend_mode == 'set':
            img[p_rr, p_cc] = v
    return downscale_local_mean(img, (smooth_sr_fact, smooth_sr_fact)) if smooth_sr_fact != 1 else img

class RectanglesDataset(torch.utils.data.IterableDataset):
    """
    Dataset with images of multiple random rectangles.
    The images are normalized to have a value range of ``[0., 1.]`` with a
    background value of ``0.``.
    """
    def __init__(self, image_size=28, num_rects=1, num_angle_modes=None, angle_modes_sigma=0.25, length=32000, fixed_seed=1, smooth_sr_fact=8):
        super().__init__()
        self.shape = (image_size, image_size)
        # defining discretization space ODL
        self.num_rects = num_rects
        self.num_angle_modes = num_angle_modes or num_rects
        self.angle_modes_sigma = angle_modes_sigma
        self.length = length
        self.fixed_seed = None if fixed_seed in [False, None] else int(fixed_seed)
        self.smooth_sr_fact = smooth_sr_fact

    def __iter__(self):
        # note: this dataset will return the same images for each worker if using multiprocessing and a fixed seed
        r = np.random.RandomState(self.fixed_seed)
        it = repeat(None, self.length) if self.length is not None else repeat(None)
        for _ in it:
            v = r.uniform(0.5, 1.0, (self.num_rects,))
            a1 = r.uniform(0.1, .8, (self.num_rects,))
            a2 = r.uniform(0.1, .8, (self.num_rects,))
            x = r.uniform(-.75, .75, (self.num_rects,))
            y = r.uniform(-.75, .75, (self.num_rects,))
            angle_modes = r.uniform(0., np.pi, (self.num_angle_modes,))
            angle_modes_per_rect = angle_modes[r.randint(0, self.num_angle_modes, (self.num_rects,))]
            rot = r.normal(angle_modes_per_rect, self.angle_modes_sigma)
            rot = np.mod(rot, np.pi)
            rects = np.stack((v, a1, a2, x, y, rot), axis=1)
            image = rect_phantom(self.shape, rects, self.smooth_sr_fact)
            # normalize the foreground (all non-zero pixels) to [0., 1.]
            image[np.array(image) != 0.] -= np.min(image)
            image /= np.max(image)
            yield image[None]
