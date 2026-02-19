"""
Multiscale design variable parameterization.

Provides MultiscaleRepresenter, which parameterizes each macroscopic voxel with:
- vf_logit: volume fraction (mapped to [vf_min, vf_max] via sigmoid)
- type_logits: TPMS type weights (mapped via softmax)

These parameters are converted to per-voxel anisotropic stiffness tensors
via a HomogenizationLookupTable with differentiable interpolation.
"""

__all__ = ['MultiscaleRepresenter']

import torch
import torch.nn as nn
import torch.nn.functional as TF

from .density_filters import RadialDensityFilter


class MultiscaleRepresenter(nn.Module):
    """Parameterizes multiscale design variables for TPMS optimization.

    Each macroscopic voxel has:
    - A volume fraction (continuous, in [vf_min, vf_max])
    - TPMS type weights (continuous relaxation via softmax)

    The forward pass returns:
    - C_field: (9, 9, nx, ny, nz) per-voxel stiffness tensor
    - vf: (1, nx, ny, nz) volume fraction field

    All operations are differentiable for end-to-end optimization.
    """

    def __init__(self, problem, homogenization_table, n_tpms_types=None,
                 initial_vf=0.3, vf_min=0.05, vf_max=0.95,
                 filter_size=3, use_filter=True,
                 temperature=1.0, dtype=torch.float32):
        """
        Parameters
        ----------
        problem : dl4to.problem.Problem
            The optimization problem.
        homogenization_table : HomogenizationLookupTable
            Precomputed lookup table for C_eff interpolation.
        n_tpms_types : int, optional
            Number of TPMS types. Defaults to table's n_types.
        initial_vf : float
            Initial volume fraction for all voxels.
        vf_min, vf_max : float
            Bounds for volume fraction.
        filter_size : int
            Size of spatial smoothing filter.
        use_filter : bool
            Whether to apply spatial filtering.
        temperature : float
            Initial softmax temperature for type selection.
        dtype : torch.dtype
        """
        super().__init__()

        self.problem = problem
        self.table = homogenization_table
        self.vf_min = vf_min
        self.vf_max = vf_max
        self.temperature = temperature
        self.dtype = dtype

        if n_tpms_types is None:
            n_tpms_types = homogenization_table.n_types
        self.n_tpms_types = n_tpms_types

        nx, ny, nz = problem.shape

        # Volume fraction logit parameter
        # Initialize so that sigmoid(logit) ≈ initial_vf
        # sigmoid(x) = initial_vf => x = log(initial_vf / (1 - initial_vf))
        vf_init = torch.log(torch.tensor(initial_vf / (1.0 - initial_vf)))
        self.vf_logit = nn.Parameter(
            vf_init * torch.ones(1, nx, ny, nz, dtype=dtype),
            requires_grad=True
        )

        # TPMS type logits (uniform initialization → equal weights after softmax)
        self.type_logits = nn.Parameter(
            torch.zeros(n_tpms_types, nx, ny, nz, dtype=dtype),
            requires_grad=True
        )

        # Spatial smoothing filter
        self.use_filter = use_filter
        if use_filter:
            self.vf_filter = RadialDensityFilter(filter_size=filter_size, dtype=dtype)

    def _sigmoid_scaled(self, logit):
        """Map logit to [vf_min, vf_max] via scaled sigmoid."""
        return self.vf_min + (self.vf_max - self.vf_min) * torch.sigmoid(logit)

    def get_volume_fraction(self):
        """Get the current volume fraction field.

        Returns
        -------
        vf : torch.Tensor, shape (1, nx, ny, nz)
            Volume fraction in [vf_min, vf_max].
        """
        logit = self.vf_logit

        # Apply spatial filter for smoothness
        if self.use_filter:
            # Filter expects (batch, 1, nx, ny, nz) → output (batch, 1, nx, ny, nz)
            logit_filtered = self.vf_filter(logit.unsqueeze(0)).squeeze(0)
        else:
            logit_filtered = logit

        vf = self._sigmoid_scaled(logit_filtered)

        # Enforce design space constraints
        Omega_design = self.problem.Ω_design.to(vf.device)
        vf = torch.where(Omega_design == 0, torch.zeros_like(vf), vf)
        vf = torch.where(Omega_design == 1, torch.ones_like(vf), vf)

        return vf

    def get_type_weights(self):
        """Get the TPMS type selection weights.

        Returns
        -------
        weights : torch.Tensor, shape (n_types, nx, ny, nz)
            Softmax weights summing to 1 along dim=0.
        """
        logits = self.type_logits / self.temperature
        return TF.softmax(logits, dim=0)

    def forward(self):
        """Compute the per-voxel stiffness field and volume fraction.

        Returns
        -------
        C_field : torch.Tensor, shape (9, 9, nx, ny, nz)
            Per-voxel anisotropic stiffness tensor.
        vf : torch.Tensor, shape (1, nx, ny, nz)
            Volume fraction field.
        """
        vf = self.get_volume_fraction()  # (1, nx, ny, nz)
        type_weights = self.get_type_weights()  # (T, nx, ny, nz)

        # Get C_eff for each type at each voxel's volume fraction
        # C_per_type: (T, 9, 9, nx, ny, nz)
        C_per_type = self.table.batch_interpolate_9x9(vf)
        C_per_type = C_per_type.to(device=vf.device, dtype=vf.dtype)

        # Weighted combination across TPMS types
        # type_weights: (T, nx, ny, nz) → (T, 1, 1, nx, ny, nz)
        w = type_weights.unsqueeze(1).unsqueeze(1)
        C_field = (w * C_per_type).sum(dim=0)  # (9, 9, nx, ny, nz)

        # Apply design space constraints: void voxels get near-zero stiffness
        C_field = self._apply_design_space(C_field, vf)

        return C_field, vf

    def _apply_design_space(self, C_field, vf):
        """Apply design space constraints to stiffness field.

        Voxels outside the design space (Omega_design == 0) get minimal stiffness.
        Voxels that must be solid (Omega_design == 1) keep their stiffness.

        Parameters
        ----------
        C_field : torch.Tensor, shape (9, 9, nx, ny, nz)
        vf : torch.Tensor, shape (1, nx, ny, nz)

        Returns
        -------
        C_field : torch.Tensor, shape (9, 9, nx, ny, nz)
        """
        Omega_design = self.problem.Ω_design.to(C_field.device)  # (1, nx, ny, nz)
        # Mask: 0 where void, 1 elsewhere
        mask = (Omega_design != 0).float()  # (1, nx, ny, nz)
        # Broadcast mask to (1, 1, nx, ny, nz) for C_field multiplication
        C_field = C_field * mask.unsqueeze(0)
        return C_field

    def get_optimized_params(self):
        """Return the optimized TPMS parameters as a dictionary.

        Returns
        -------
        params : dict
            Contains 'vf', 'type_weights', 'type_logits', 'vf_logit'.
        """
        with torch.no_grad():
            vf = self.get_volume_fraction()
            type_weights = self.get_type_weights()

        return {
            'vf': vf.detach(),
            'type_weights': type_weights.detach(),
            'vf_logit': self.vf_logit.detach().clone(),
            'type_logits': self.type_logits.detach().clone(),
            'tpms_types': self.table.tpms_types,
        }

    def set_temperature(self, temperature):
        """Update the softmax temperature for TPMS type selection.

        Lower temperature → sharper selection (more discrete).
        """
        self.temperature = temperature
