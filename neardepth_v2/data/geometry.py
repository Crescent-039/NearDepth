"""Read-time conversion of Hypersim camera-ray distances to positive axial z."""
from functools import lru_cache
import numpy as np


def validate_geometry(metadata):
    convention = metadata.get('depth_convention')
    if convention == 'z':
        return
    if convention != 'radial':
        raise ValueError('depth_convention must explicitly be z or radial')
    matrix = np.asarray(metadata.get('camera_ray_matrix'), dtype=np.float64)
    size = metadata.get('camera_native_size', [])
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError('radial depth requires a finite invertible camera_ray_matrix')
    if len(size) != 2 or any(not isinstance(v, int) or v <= 0 for v in size):
        raise ValueError('radial depth requires camera_native_size=[height,width]')


@lru_cache(maxsize=8)
def radial_z_factor(height, width, matrix_values):
    matrix = np.asarray(matrix_values, dtype=np.float64).reshape(3, 3)
    # Official Hypersim convention: NDC pixel centers, v positive upwards,
    # OpenGL cameras looking down -z. The per-scene matrix includes tilt/shift.
    u = (2*(np.arange(width, dtype=np.float64)+.5)/width-1)[None, :]
    v = (1-2*(np.arange(height, dtype=np.float64)+.5)/height)[:, None]
    x = matrix[0,0]*u+matrix[0,1]*v+matrix[0,2]
    y = matrix[1,0]*u+matrix[1,1]*v+matrix[1,2]
    z = matrix[2,0]*u+matrix[2,1]*v+matrix[2,2]
    norm = np.sqrt(x*x+y*y+z*z)
    if not np.isfinite(norm).all() or np.any(norm <= 0) or np.any(z >= 0):
        raise ValueError('Hypersim camera rays must be finite and point along negative z')
    factor = np.broadcast_to(-z/norm, (height, width)).astype(np.float32).copy()
    factor.flags.writeable = False
    return factor


def to_axial_depth(depth, metadata):
    validate_geometry(metadata)
    if metadata['depth_convention'] == 'z':
        return depth
    if tuple(metadata['camera_native_size']) != depth.shape:
        raise ValueError('radial depth resolution differs from camera metadata')
    matrix = np.asarray(metadata['camera_ray_matrix'], dtype=np.float64)
    factor = radial_z_factor(*depth.shape, tuple(matrix.ravel()))
    # Invalid NaN/zero/nonpositive values remain invalid for the normal loader mask.
    return depth*factor
