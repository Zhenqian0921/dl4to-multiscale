"""
TPMS (Triply Periodic Minimal Surface) geometry module.

Provides differentiable level-set functions for 6 common TPMS types,
along with a voxelization function using smooth sigmoid approximation.
All operations are PyTorch-based and support automatic differentiation.
"""

__all__ = [
    'TPMSFunction', 'Gyroid', 'SchwarzP', 'SchwarzD',
    'Neovius', 'FischerKoch', 'IWP',
    'TPMS_REGISTRY', 'tpms_density_field', 'compute_volume_fraction',
    'find_threshold_for_vf',
]

import torch
import math


class TPMSFunction:
    """Base class for TPMS level-set functions."""
    name: str = "base"

    def __call__(self, x, y, z, period=1.0):
        """Evaluate the TPMS level-set function.

        Parameters
        ----------
        x, y, z : torch.Tensor
            Coordinate tensors (same shape).
        period : float
            Spatial period of the TPMS.

        Returns
        -------
        torch.Tensor
            Level-set values.
        """
        raise NotImplementedError


class Gyroid(TPMSFunction):
    """Gyroid: sin(2*pi*x/L)*cos(2*pi*y/L) + sin(2*pi*y/L)*cos(2*pi*z/L) + sin(2*pi*z/L)*cos(2*pi*x/L)"""
    name = "gyroid"

    def __call__(self, x, y, z, period=1.0):
        k = 2.0 * math.pi / period
        return (torch.sin(k * x) * torch.cos(k * y)
                + torch.sin(k * y) * torch.cos(k * z)
                + torch.sin(k * z) * torch.cos(k * x))


class SchwarzP(TPMSFunction):
    """Schwarz Primitive: cos(2*pi*x/L) + cos(2*pi*y/L) + cos(2*pi*z/L)"""
    name = "schwarz_p"

    def __call__(self, x, y, z, period=1.0):
        k = 2.0 * math.pi / period
        return torch.cos(k * x) + torch.cos(k * y) + torch.cos(k * z)


class SchwarzD(TPMSFunction):
    """Schwarz Diamond:
    sin(kx)*sin(ky)*sin(kz) + sin(kx)*cos(ky)*cos(kz)
    + cos(kx)*sin(ky)*cos(kz) + cos(kx)*cos(ky)*sin(kz)
    """
    name = "schwarz_d"

    def __call__(self, x, y, z, period=1.0):
        k = 2.0 * math.pi / period
        sx, sy, sz = torch.sin(k * x), torch.sin(k * y), torch.sin(k * z)
        cx, cy, cz = torch.cos(k * x), torch.cos(k * y), torch.cos(k * z)
        return (sx * sy * sz
                + sx * cy * cz
                + cx * sy * cz
                + cx * cy * sz)


class Neovius(TPMSFunction):
    """Neovius: 3*(cos(kx) + cos(ky) + cos(kz)) + 4*cos(kx)*cos(ky)*cos(kz)"""
    name = "neovius"

    def __call__(self, x, y, z, period=1.0):
        k = 2.0 * math.pi / period
        cx, cy, cz = torch.cos(k * x), torch.cos(k * y), torch.cos(k * z)
        return 3.0 * (cx + cy + cz) + 4.0 * cx * cy * cz


class FischerKoch(TPMSFunction):
    """Fischer-Koch S:
    cos(2kx)*sin(ky)*cos(kz) + cos(kx)*cos(2ky)*sin(kz)
    + sin(kx)*cos(ky)*cos(2kz)
    """
    name = "fischer_koch"

    def __call__(self, x, y, z, period=1.0):
        k = 2.0 * math.pi / period
        sx, sy, sz = torch.sin(k * x), torch.sin(k * y), torch.sin(k * z)
        cx, cy, cz = torch.cos(k * x), torch.cos(k * y), torch.cos(k * z)
        c2x = torch.cos(2.0 * k * x)
        c2y = torch.cos(2.0 * k * y)
        c2z = torch.cos(2.0 * k * z)
        return c2x * sy * cz + cx * c2y * sz + sx * cy * c2z


class IWP(TPMSFunction):
    """IWP (I-graph and Wrapped Package):
    cos(kx)*cos(ky) + cos(ky)*cos(kz) + cos(kz)*cos(kx)
    - cos(kx)*cos(ky)*cos(kz)
    """
    name = "iwp"

    def __call__(self, x, y, z, period=1.0):
        k = 2.0 * math.pi / period
        cx, cy, cz = torch.cos(k * x), torch.cos(k * y), torch.cos(k * z)
        return (cx * cy + cy * cz + cz * cx
                - cx * cy * cz)


TPMS_REGISTRY = {
    'gyroid': Gyroid,
    'schwarz_p': SchwarzP,
    'schwarz_d': SchwarzD,
    'neovius': Neovius,
    'fischer_koch': FischerKoch,
    'iwp': IWP,
}


def _make_unit_cell_grid(resolution, period=1.0, device='cpu', dtype=torch.float32):
    """Create a 3D coordinate grid for one TPMS unit cell.

    Parameters
    ----------
    resolution : int or tuple of int
        Number of voxels per axis.
    period : float
        Spatial period of the TPMS.
    device : str or torch.device
    dtype : torch.dtype

    Returns
    -------
    x, y, z : torch.Tensor
        Coordinate tensors of shape (res, res, res).
    """
    if isinstance(resolution, int):
        resolution = (resolution, resolution, resolution)
    rx, ry, rz = resolution

    lx = torch.linspace(0, period, rx + 1, device=device, dtype=dtype)[:-1]
    ly = torch.linspace(0, period, ry + 1, device=device, dtype=dtype)[:-1]
    lz = torch.linspace(0, period, rz + 1, device=device, dtype=dtype)[:-1]

    # Shift to cell centers
    lx = lx + period / (2.0 * rx)
    ly = ly + period / (2.0 * ry)
    lz = lz + period / (2.0 * rz)

    x, y, z = torch.meshgrid(lx, ly, lz, indexing='ij')
    return x, y, z


def tpms_density_field(tpms_func, grid_shape, period=1.0, threshold=0.0,
                       smooth_width=0.1, device='cpu', dtype=torch.float32):
    """Generate a differentiable TPMS density field using sigmoid approximation.

    The density at each voxel is: sigmoid((phi(x,y,z) - threshold) / smooth_width)
    where phi is the TPMS level-set function.

    Parameters
    ----------
    tpms_func : TPMSFunction
        An instance of a TPMS function class.
    grid_shape : int or tuple of int
        Resolution per axis.
    period : float
        Spatial period.
    threshold : float
        Level-set threshold controlling volume fraction.
    smooth_width : float
        Controls transition sharpness (smaller = sharper).
    device : str or torch.device
    dtype : torch.dtype

    Returns
    -------
    density : torch.Tensor
        Density field of shape (1, nx, ny, nz) with values in [0, 1].
    """
    x, y, z = _make_unit_cell_grid(grid_shape, period, device, dtype)
    phi = tpms_func(x, y, z, period)
    density = torch.sigmoid((phi - threshold) / smooth_width)
    return density.unsqueeze(0)  # (1, nx, ny, nz)


def compute_volume_fraction(density_field):
    """Compute the volume fraction of a density field.

    Parameters
    ----------
    density_field : torch.Tensor
        Shape (1, nx, ny, nz) or (nx, ny, nz).

    Returns
    -------
    float
        Volume fraction (mean density).
    """
    return density_field.mean().item()


def find_threshold_for_vf(tpms_func, target_vf, grid_shape=32, period=1.0,
                          smooth_width=0.05, tol=1e-4, max_iter=100,
                          device='cpu', dtype=torch.float32):
    """Binary search for the threshold that achieves a target volume fraction.

    Parameters
    ----------
    tpms_func : TPMSFunction
    target_vf : float
        Target volume fraction in (0, 1).
    grid_shape : int
    period : float
    smooth_width : float
    tol : float
    max_iter : int
    device : str
    dtype : torch.dtype

    Returns
    -------
    threshold : float
    """
    x, y, z = _make_unit_cell_grid(grid_shape, period, device, dtype)
    phi = tpms_func(x, y, z, period)

    lo, hi = phi.min().item() - 1.0, phi.max().item() + 1.0

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        density = torch.sigmoid((phi - mid) / smooth_width)
        vf = density.mean().item()

        if abs(vf - target_vf) < tol:
            return mid

        if vf > target_vf:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0
