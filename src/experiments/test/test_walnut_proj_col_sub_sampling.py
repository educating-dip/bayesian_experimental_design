import numpy as np
import matplotlib.pyplot as plt

from dataset.walnuts_interface import (
        WalnutRayTrafo, get_first_proj_col_for_sub_sampling)

angular_sub_sampling = 20
proj_col_sub_sampling = 6
first_proj_col = -1

if first_proj_col == -1:
    first_proj_col = get_first_proj_col_for_sub_sampling(
            factor=proj_col_sub_sampling)

print('first_proj_col', first_proj_col)

walnut_ray_trafo = WalnutRayTrafo(
        data_path='walnuts', walnut_id=1, orbit_id=2,
        angular_sub_sampling=angular_sub_sampling,
        proj_col_sub_sampling=1)
walnut_ray_trafo_css = WalnutRayTrafo(
        data_path='walnuts', walnut_id=1, orbit_id=2,
        angular_sub_sampling=angular_sub_sampling,
        proj_col_sub_sampling=proj_col_sub_sampling,
        proj_col_sub_sampling_via_geom=True,  # changing to False yields 0 diff
        first_proj_col=first_proj_col,
        )

np.random.seed(1)

x = np.random.random(walnut_ray_trafo.vol_shape)

y = walnut_ray_trafo.fp3d(x)
# y_post_css = y.reshape(  # downscale
#         y.shape[0], y.shape[1], y.shape[2] // proj_col_sub_sampling,
#         proj_col_sub_sampling).mean(axis=-1)
y_post_css = y[:, :, first_proj_col::proj_col_sub_sampling]

y_css = walnut_ray_trafo_css.fp3d(x)

print(y_css.shape)

print('max(abs(y_post_css-y_css))', np.max(np.abs(y_post_css-y_css)))
print('mean(abs(y_post_css-y_css))', np.mean(np.abs(y_post_css-y_css)))
print('mean(abs(y_post_css))', np.mean(np.abs(y_post_css)))
print('mean(abs(y_css))', np.mean(np.abs(y_css)))

plt.subplot(1, 3, 1)
plt.imshow(y_post_css[y_post_css.shape[0] // 2, :, :])
plt.subplot(1, 3, 2)
plt.imshow(y_css[y_css.shape[0] // 2, :, :])
plt.subplot(1, 3, 3)
plt.imshow((y_post_css - y_css)[y_css.shape[0] // 2, :, :])

plt.show()
