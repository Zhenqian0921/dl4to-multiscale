"""
High-resolution TPMS structure reconstruction.

Converts macroscale optimization results (per-voxel TPMS type + volume fraction)
into a high-resolution voxel model where each macroscopic voxel is replaced
by a detailed TPMS unit cell at the specified volume fraction.
"""

__all__ = ['TPMSReconstructor']

import torch
import torch.nn.functional as TF

from .tpms import TPMS_REGISTRY, tpms_density_field, find_threshold_for_vf, _make_unit_cell_grid


class TPMSReconstructor:
    """Reconstructs high-resolution TPMS structures from macroscale parameters.

    Each macroscopic voxel is expanded to upscale_factor^3 micro-voxels
    containing the appropriate TPMS geometry at the specified volume fraction.
    """

    def __init__(self, upscale_factor=8, smooth_width=0.05,
                 boundary_smooth_sigma=1.0, device='cpu',
                 dtype=torch.float32):
        """
        Parameters
        ----------
        upscale_factor : int
            Number of micro-voxels per macro-voxel edge.
        smooth_width : float
            Sigmoid sharpness for TPMS density field generation.
        boundary_smooth_sigma : float
            Gaussian smoothing sigma for cell boundary transitions.
            Set to 0 to disable boundary smoothing.
        device : str or torch.device
        dtype : torch.dtype
        """
        self.upscale_factor = upscale_factor
        self.smooth_width = smooth_width
        self.boundary_smooth_sigma = boundary_smooth_sigma
        self.device = device
        self.dtype = dtype

    def _get_dominant_tpms(self, type_weights):
        """Get the dominant TPMS type index at each voxel.

        Parameters
        ----------
        type_weights : torch.Tensor, shape (T, nx, ny, nz)

        Returns
        -------
        indices : torch.Tensor, shape (nx, ny, nz), dtype=long
        """
        return type_weights.argmax(dim=0)

    def _find_threshold(self, tpms_func, target_vf):
        """Find threshold for a given TPMS at target volume fraction."""
        return find_threshold_for_vf(
            tpms_func, target_vf,
            grid_shape=self.upscale_factor,
            smooth_width=self.smooth_width,
            device=self.device, dtype=self.dtype
        )

    def _generate_unit_cell(self, tpms_func, threshold):
        """Generate a single TPMS unit cell density field.

        Parameters
        ----------
        tpms_func : TPMSFunction instance
        threshold : float

        Returns
        -------
        cell : torch.Tensor, shape (uf, uf, uf)
        """
        density = tpms_density_field(
            tpms_func, self.upscale_factor,
            period=1.0, threshold=threshold,
            smooth_width=self.smooth_width,
            device=self.device, dtype=self.dtype
        )
        return density.squeeze(0)  # (uf, uf, uf)

    def _gaussian_smooth_3d(self, field, sigma):
        """Apply 3D Gaussian smoothing.

        Parameters
        ----------
        field : torch.Tensor, shape (1, 1, nx, ny, nz)
        sigma : float

        Returns
        -------
        smoothed : torch.Tensor, same shape
        """
        if sigma <= 0:
            return field

        # Create 1D Gaussian kernel
        kernel_size = max(3, int(6 * sigma + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = torch.arange(kernel_size, dtype=self.dtype, device=self.device)
        x = x - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Apply separable 3D convolution
        pad = kernel_size // 2

        # X direction
        k = kernel_1d.reshape(1, 1, -1, 1, 1)
        field = TF.pad(field, (0, 0, 0, 0, pad, pad), mode='replicate')
        field = TF.conv3d(field, k)

        # Y direction
        k = kernel_1d.reshape(1, 1, 1, -1, 1)
        field = TF.pad(field, (0, 0, pad, pad, 0, 0), mode='replicate')
        field = TF.conv3d(field, k)

        # Z direction
        k = kernel_1d.reshape(1, 1, 1, 1, -1)
        field = TF.pad(field, (pad, pad, 0, 0, 0, 0), mode='replicate')
        field = TF.conv3d(field, k)

        return field

    def reconstruct(self, problem, tpms_params):
        """Reconstruct high-resolution TPMS structure.

        Parameters
        ----------
        problem : dl4to.problem.Problem
        tpms_params : dict
            Must contain:
            - 'vf': torch.Tensor, shape (1, nx, ny, nz)
            - 'type_weights': torch.Tensor, shape (T, nx, ny, nz)
            - 'tpms_types': list of str

        Returns
        -------
        hr_density : torch.Tensor, shape (1, nx*uf, ny*uf, nz*uf)
            High-resolution binary-ish density field.
        """
        vf = tpms_params['vf'].to(device=self.device, dtype=self.dtype)
        type_weights = tpms_params['type_weights'].to(device=self.device, dtype=self.dtype)
        tpms_types = tpms_params['tpms_types']

        nx, ny, nz = vf.shape[1:]
        uf = self.upscale_factor

        # Get dominant TPMS type per voxel
        dominant_type = self._get_dominant_tpms(type_weights)  # (nx, ny, nz)

        # Instantiate TPMS functions
        tpms_funcs = [TPMS_REGISTRY[t]() for t in tpms_types]

        # Pre-cache: for each (type_idx, discretized_vf) → unit cell
        # Discretize vf to reduce redundant computation
        vf_values = vf.squeeze(0)  # (nx, ny, nz)
        n_vf_levels = 50  # discretize to 50 levels
        vf_discrete = (vf_values * n_vf_levels).round() / n_vf_levels
        vf_discrete = vf_discrete.clamp(0.02, 0.98)

        # Build cache
        cell_cache = {}

        # Allocate output
        hr_density = torch.zeros(nx * uf, ny * uf, nz * uf,
                                 device=self.device, dtype=self.dtype)

        # Fill each macro voxel
        design_space = problem.Ω_design.to(self.device)  # (1, nx, ny, nz)

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    ds_val = design_space[0, ix, iy, iz].item()

                    # Void voxels
                    if ds_val == 0:
                        continue

                    # Solid voxels
                    if ds_val == 1:
                        hr_density[
                            ix * uf:(ix + 1) * uf,
                            iy * uf:(iy + 1) * uf,
                            iz * uf:(iz + 1) * uf
                        ] = 1.0
                        continue

                    # Design voxels: fill with TPMS unit cell
                    t_idx = dominant_type[ix, iy, iz].item()
                    vf_val = vf_discrete[ix, iy, iz].item()

                    cache_key = (t_idx, round(vf_val, 3))
                    if cache_key not in cell_cache:
                        tpms_func = tpms_funcs[t_idx]
                        threshold = self._find_threshold(tpms_func, vf_val)
                        cell = self._generate_unit_cell(tpms_func, threshold)
                        cell_cache[cache_key] = cell

                    cell = cell_cache[cache_key]
                    hr_density[
                        ix * uf:(ix + 1) * uf,
                        iy * uf:(iy + 1) * uf,
                        iz * uf:(iz + 1) * uf
                    ] = cell

        # Optional boundary smoothing
        if self.boundary_smooth_sigma > 0:
            hr_5d = hr_density.unsqueeze(0).unsqueeze(0)
            hr_5d = self._gaussian_smooth_3d(hr_5d, self.boundary_smooth_sigma)
            hr_density = hr_5d.squeeze(0).squeeze(0)

        # Clamp to [0, 1]
        hr_density = hr_density.clamp(0, 1)

        return hr_density.unsqueeze(0)  # (1, nx*uf, ny*uf, nz*uf)

    def reconstruct_binary(self, problem, tpms_params, threshold=0.5):
        """Reconstruct and binarize high-resolution structure.

        Parameters
        ----------
        problem : dl4to.problem.Problem
        tpms_params : dict
        threshold : float
            Binarization threshold.

        Returns
        -------
        hr_binary : torch.Tensor, shape (1, nx*uf, ny*uf, nz*uf)
        """
        hr_density = self.reconstruct(problem, tpms_params)
        hr_binary = (hr_density > threshold).float()
        return hr_binary

    def compute_reconstructed_vf(self, problem, tpms_params):
        """Compute the actual volume fraction of the reconstructed structure.

        Parameters
        ----------
        problem : dl4to.problem.Problem
        tpms_params : dict

        Returns
        -------
        vf : float
        """
        hr_density = self.reconstruct(problem, tpms_params)
        return hr_density.mean().item()
