"""
Numerical homogenization module for computing effective elastic stiffness tensors
of TPMS unit cells.

Provides:
- PeriodicFDMSolver: FDM solver with periodic boundary conditions for unit cell problems.
- NumericalHomogenizer: Computes C_eff (6x6 Voigt) from a density field.
- HomogenizationLookupTable: Precomputes and interpolates C_eff as a function
  of TPMS type and volume fraction.
"""

__all__ = [
    'PeriodicFDMSolver', 'NumericalHomogenizer', 'HomogenizationLookupTable',
]

import torch
import torch.nn.functional as F
import numpy as np
import math
import os
from scipy.sparse.linalg import factorized, spsolve
from scipy.sparse import csc_matrix, diags, eye as speye

from .tpms import TPMS_REGISTRY, tpms_density_field, find_threshold_for_vf


# ============================================================================
# Voigt notation utilities
# ============================================================================

# 6 canonical macroscopic strain modes in Voigt notation:
# [e_xx, e_yy, e_zz, 2*e_yz, 2*e_xz, 2*e_xy]
# Each is a (6,) vector; there are 6 of them.
_VOIGT_STRAIN_MODES = torch.eye(6)

# Map from Voigt index to (i, j) pairs in 3x3 symmetric tensor
_VOIGT_MAP = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

# Map from 9-component tensor to Voigt: index in 9-channel → Voigt index
# 9-channel layout from pde.py: [du0/dx0, du0/dx1, du0/dx2, du1/dx0, du1/dx1, du1/dx2, du2/dx0, du2/dx1, du2/dx2]
# i.e., sigma[k] = sigma[i][j] where i=k//3, j=k%3
# Voigt: [11, 22, 33, 23, 13, 12] → 9-idx: [0, 4, 8, 5, 2, 1]
_NINE_TO_VOIGT_IDX = [0, 4, 8, 5, 2, 1]
_VOIGT_TO_NINE_IDX = [0, 5, 4, 5, 1, 3, 4, 3, 2]  # symmetric mapping


def _voigt_strain_to_9(voigt_strain):
    """Convert a 6-component Voigt strain to 9-component tensor strain.

    Voigt: [e11, e22, e33, 2*e23, 2*e13, 2*e12]
    9-comp: [e11, e12, e13, e21, e22, e23, e31, e32, e33]

    For strain: off-diagonal Voigt components contain 2*e_ij,
    so e_ij = voigt[k]/2 for k >= 3.
    """
    e = torch.zeros(9, dtype=voigt_strain.dtype, device=voigt_strain.device)
    # Diagonal
    e[0] = voigt_strain[0]  # e11
    e[4] = voigt_strain[1]  # e22
    e[8] = voigt_strain[2]  # e33
    # Off-diagonal (symmetric, divide by 2 because Voigt uses 2*e_ij)
    e[5] = voigt_strain[3] / 2.0  # e23
    e[7] = voigt_strain[3] / 2.0  # e32
    e[2] = voigt_strain[4] / 2.0  # e13
    e[6] = voigt_strain[4] / 2.0  # e31
    e[1] = voigt_strain[5] / 2.0  # e12
    e[3] = voigt_strain[5] / 2.0  # e21
    return e


def _stress_9_to_voigt(stress_9):
    """Convert 9-component stress to 6-component Voigt stress.

    9-comp: [s11, s12, s13, s21, s22, s23, s31, s32, s33]
    Voigt: [s11, s22, s33, s23, s13, s12]
    """
    v = torch.zeros(6, dtype=stress_9.dtype, device=stress_9.device)
    v[0] = stress_9[0]  # s11
    v[1] = stress_9[4]  # s22
    v[2] = stress_9[8]  # s33
    v[3] = stress_9[5]  # s23
    v[4] = stress_9[2]  # s13
    v[5] = stress_9[1]  # s12
    return v


def _get_isotropic_C_voigt(E, nu):
    """Build 6x6 isotropic stiffness matrix in Voigt notation.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.

    Returns
    -------
    C : torch.Tensor, shape (6, 6)
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    C = torch.zeros(6, 6)
    # Diagonal
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    # Off-diagonal
    C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
    return C


def _get_G_matrix_9x9(E, nu):
    """Build the 9x9 constitutive matrix used by the existing FDM solver.

    This is the G matrix from UnpaddedFDM._get_G, normalized by E.
    """
    factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    G = torch.tensor([
        [1 - nu, 0, 0, 0, nu, 0, 0, 0, nu],
        [0, 0.5 - nu, 0, 0.5 - nu, 0, 0, 0, 0, 0],
        [0, 0, 0.5 - nu, 0, 0, 0, 0.5 - nu, 0, 0],

        [0, 0.5 - nu, 0, 0.5 - nu, 0, 0, 0, 0, 0],
        [nu, 0, 0, 0, 1 - nu, 0, 0, 0, nu],
        [0, 0, 0, 0, 0, 0.5 - nu, 0, 0.5 - nu, 0],

        [0, 0, 0.5 - nu, 0, 0, 0, 0.5 - nu, 0, 0],
        [0, 0, 0, 0, 0, 0.5 - nu, 0, 0.5 - nu, 0],
        [nu, 0, 0, 0, nu, 0, 0, 0, 1 - nu]
    ], dtype=torch.float64) * factor
    return G


# ============================================================================
# Periodic FDM derivatives
# ============================================================================

def _du_dx_periodic(u, h):
    """Forward-difference derivative along x with periodic wrapping."""
    # u: (C, nx, ny, nz)
    du = (torch.roll(u, -1, dims=1) - u) / h
    return du


def _du_dy_periodic(u, h):
    du = (torch.roll(u, -1, dims=2) - u) / h
    return du


def _du_dz_periodic(u, h):
    du = (torch.roll(u, -1, dims=3) - u) / h
    return du


def _du_dx_adj_periodic(eps, h):
    """Adjoint of forward-difference along x with periodic BC."""
    return (torch.roll(eps, 1, dims=1) - eps) / h


def _du_dy_adj_periodic(eps, h):
    return (torch.roll(eps, 1, dims=2) - eps) / h


def _du_dz_adj_periodic(eps, h):
    return (torch.roll(eps, 1, dims=3) - eps) / h


# ============================================================================
# Periodic FDM Solver for unit cell homogenization
# ============================================================================

class PeriodicFDMSolver:
    """FDM solver with periodic boundary conditions for unit cell problems.

    Solves -div(C:eps(u)) = f on a periodic unit cell using finite differences
    with forward differences and periodic wrapping via torch.roll.
    """

    def __init__(self, shape, h, G_matrix, device='cpu', dtype=torch.float64):
        """
        Parameters
        ----------
        shape : tuple of int
            (nx, ny, nz) grid resolution.
        h : float or list of float
            Voxel size(s).
        G_matrix : torch.Tensor, shape (9, 9)
            Constitutive matrix (constant for unit cell of given material).
        device : str
        dtype : torch.dtype
        """
        self.shape = shape
        self.device = device
        self.dtype = dtype

        if isinstance(h, (int, float)):
            self.h = [float(h)] * 3
        else:
            self.h = [float(hi) for hi in h]

        self.G = G_matrix.to(device=device, dtype=dtype)
        self.ndof = 3 * np.prod(shape)

    def _J(self, u):
        """Strain operator: u (3, nx, ny, nz) -> eps (9, nx, ny, nz)."""
        eps = torch.cat([
            _du_dx_periodic(u, self.h[0]),
            _du_dy_periodic(u, self.h[1]),
            _du_dz_periodic(u, self.h[2]),
        ], dim=0)
        return eps

    def _J_adj(self, sigma):
        """Adjoint of strain operator: sigma (9, nx, ny, nz) -> u (3, nx, ny, nz)."""
        return (_du_dx_adj_periodic(sigma[:3], self.h[0])
                + _du_dy_adj_periodic(sigma[3:6], self.h[1])
                + _du_dz_adj_periodic(sigma[6:], self.h[2]))

    def _apply_constitutive(self, eps, density):
        """Apply constitutive relation with local density scaling.

        sigma_i(x) = density(x) * G_ij * eps_j(x)
        """
        sigma = torch.einsum('ij, jxyz -> ixyz', self.G, eps)
        # Scale by local density (SIMP with p=1 for homogenization)
        sigma = density * sigma
        return sigma

    def _A_op(self, u, density):
        """System operator: A(u) = J^T (density * G * J(u))."""
        eps = self._J(u)
        sigma = self._apply_constitutive(eps, density)
        return self._J_adj(sigma)

    def _assemble_A_matrix(self, density):
        """Assemble sparse system matrix for the periodic problem.

        Uses column-by-column probing.
        """
        nx, ny, nz = self.shape
        n = 3 * nx * ny * nz

        rows = []
        cols = []
        vals = []

        for dof in range(n):
            e = torch.zeros(3, nx, ny, nz, device=self.device, dtype=self.dtype)
            c = dof // (nx * ny * nz)
            rem = dof % (nx * ny * nz)
            i = rem // (ny * nz)
            jk = rem % (ny * nz)
            j = jk // nz
            k = jk % nz
            e[c, i, j, k] = 1.0

            Ae = self._A_op(e, density).flatten()
            nz_idx = torch.nonzero(Ae, as_tuple=True)[0]

            for idx in nz_idx:
                rows.append(idx.item())
                cols.append(dof)
                vals.append(Ae[idx].item())

        A = csc_matrix((vals, (rows, cols)), shape=(n, n))
        return A

    def _assemble_A_matrix_fast(self, density):
        """Faster assembly using batched unit vectors for small grids."""
        nx, ny, nz = self.shape
        n = 3 * nx * ny * nz

        # For small problems, column-by-column is fine
        if n <= 5000:
            return self._assemble_A_matrix(density)

        # For larger problems, use a stride-based approach
        # Process DOFs in batches
        batch_size = min(n, 100)
        rows_all, cols_all, vals_all = [], [], []

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            for dof in range(start, end):
                e = torch.zeros(3, nx, ny, nz, device=self.device, dtype=self.dtype)
                c = dof // (nx * ny * nz)
                rem = dof % (nx * ny * nz)
                i = rem // (ny * nz)
                jk = rem % (ny * nz)
                j = jk // nz
                k = jk % nz
                e[c, i, j, k] = 1.0

                Ae = self._A_op(e, density).flatten()
                nz_mask = Ae.abs() > 1e-15
                nz_idx = torch.nonzero(nz_mask, as_tuple=True)[0]

                for idx in nz_idx:
                    rows_all.append(idx.item())
                    cols_all.append(dof)
                    vals_all.append(Ae[idx].item())

        A = csc_matrix((vals_all, (rows_all, cols_all)), shape=(n, n))
        return A

    def solve(self, density, rhs):
        """Solve the periodic linear system A*u = rhs.

        The system is singular (rigid body modes). We fix one DOF to
        remove the singularity, equivalent to prescribing zero mean.

        Parameters
        ----------
        density : torch.Tensor, shape (1, nx, ny, nz)
            Local density field.
        rhs : torch.Tensor, shape (3, nx, ny, nz)
            Right-hand side (body force).

        Returns
        -------
        u : torch.Tensor, shape (3, nx, ny, nz)
        """
        density_expanded = density.to(device=self.device, dtype=self.dtype)

        A = self._assemble_A_matrix_fast(density_expanded)

        # Pin one DOF per displacement component to remove rigid body modes
        nx, ny, nz = self.shape
        n = 3 * nx * ny * nz
        pin_dofs = [0, nx * ny * nz, 2 * nx * ny * nz]

        # Modify A: set pinned rows/cols to identity
        for dof in pin_dofs:
            A[dof, :] = 0
            A[:, dof] = 0
            A[dof, dof] = 1.0
        A = csc_matrix(A)

        b = rhs.to(device=self.device, dtype=self.dtype).flatten().cpu().numpy()
        for dof in pin_dofs:
            b[dof] = 0.0

        try:
            solve = factorized(A)
            x = solve(b)
        except Exception:
            x = spsolve(A, b)

        u = torch.from_numpy(x).to(device=self.device, dtype=self.dtype)
        u = u.reshape(3, nx, ny, nz)
        return u


# ============================================================================
# Numerical Homogenizer
# ============================================================================

class NumericalHomogenizer:
    """Compute the effective stiffness tensor C_eff for a unit cell density field.

    Uses the strain energy approach:
    C_eff_pq = (1/|V|) * integral_V (E^p + eps^p)^T : C : (E^q + eps^q) dV

    where E^q is the q-th canonical Voigt strain mode and eps^q(x) is the
    fluctuation strain from solving the unit cell problem.
    """

    def __init__(self, E=1.0, nu=0.3, density_min=1e-4, device='cpu', dtype=torch.float64):
        """
        Parameters
        ----------
        E : float
            Young's modulus of the base (solid) material.
        nu : float
            Poisson's ratio.
        density_min : float
            Minimum density to prevent singular systems (like theta_min in SIMP).
        device : str
        dtype : torch.dtype
        """
        self.E = E
        self.nu = nu
        self.density_min = density_min
        self.device = device
        self.dtype = dtype
        self.G = _get_G_matrix_9x9(E, nu).to(device=device, dtype=dtype)

    def homogenize(self, density_field, resolution=None):
        """Compute C_eff (6x6 Voigt) for a given density field.

        Parameters
        ----------
        density_field : torch.Tensor, shape (1, nx, ny, nz)
            Density distribution of the unit cell.
        resolution : int, optional
            If provided, ignored (uses density_field shape directly).

        Returns
        -------
        C_eff : torch.Tensor, shape (6, 6)
            Effective stiffness matrix in Voigt notation.
        """
        density = density_field.to(device=self.device, dtype=self.dtype)
        if density.dim() == 3:
            density = density.unsqueeze(0)

        # Apply minimum density to prevent singular systems
        density = density.clamp(min=self.density_min)

        nx, ny, nz = density.shape[1:]
        h = 1.0 / max(nx, ny, nz)  # normalized unit cell

        solver = PeriodicFDMSolver(
            shape=(nx, ny, nz), h=h, G_matrix=self.G,
            device=self.device, dtype=self.dtype
        )

        volume = nx * ny * nz
        C_eff = torch.zeros(6, 6, device=self.device, dtype=self.dtype)

        # Store total strain fields for each load case
        total_strains = []

        for q in range(6):
            # Macro strain mode in 9-component form
            E_voigt = _VOIGT_STRAIN_MODES[q].to(device=self.device, dtype=self.dtype)
            E_9 = _voigt_strain_to_9(E_voigt)

            # Uniform macro strain field: (9, nx, ny, nz)
            E_field = E_9.reshape(9, 1, 1, 1).expand(9, nx, ny, nz)

            # RHS = -J^T(density * G * E_macro)
            sigma_macro = solver._apply_constitutive(E_field, density)
            rhs = -solver._J_adj(sigma_macro)

            # Solve for fluctuation displacement
            chi = solver.solve(density, rhs)

            # Fluctuation strain
            eps_chi = solver._J(chi)

            # Total strain = macro + fluctuation
            total_strain = E_field + eps_chi
            total_strains.append(total_strain)

        # Compute C_eff_pq = (1/V) * sum_x density(x) * (total_p)^T * G * (total_q)
        for p in range(6):
            for q in range(p, 6):
                # sigma_q = density * G * total_strain_q
                sigma_q = solver._apply_constitutive(total_strains[q], density)
                # C_eff_pq = (1/V) * sum over all voxels of total_p . sigma_q
                val = torch.sum(total_strains[p] * sigma_q) / volume
                C_eff[p, q] = val
                C_eff[q, p] = val

        return C_eff

    def homogenize_tpms(self, tpms_name, volume_fraction, resolution=16):
        """Homogenize a specific TPMS type at a given volume fraction.

        Parameters
        ----------
        tpms_name : str
            Key in TPMS_REGISTRY.
        volume_fraction : float
            Target volume fraction.
        resolution : int
            Voxel resolution per axis.

        Returns
        -------
        C_eff : torch.Tensor, shape (6, 6)
        """
        tpms_cls = TPMS_REGISTRY[tpms_name]
        tpms_func = tpms_cls()

        threshold = find_threshold_for_vf(
            tpms_func, volume_fraction, grid_shape=resolution,
            smooth_width=0.02, device=self.device, dtype=self.dtype
        )

        density = tpms_density_field(
            tpms_func, resolution, period=1.0, threshold=threshold,
            smooth_width=0.02, device=self.device, dtype=self.dtype
        )

        return self.homogenize(density)


# ============================================================================
# Lookup Table
# ============================================================================

class HomogenizationLookupTable:
    """Precomputed lookup table for C_eff as a function of TPMS type and volume fraction.

    Supports differentiable linear interpolation for use in optimization.
    """

    def __init__(self, tpms_types=None, vf_min=0.05, vf_max=0.95,
                 n_samples=19, resolution=16, E=1.0, nu=0.3,
                 device='cpu', dtype=torch.float64):
        """
        Parameters
        ----------
        tpms_types : list of str, optional
            TPMS types to include. Defaults to all in TPMS_REGISTRY.
        vf_min, vf_max : float
            Volume fraction range.
        n_samples : int
            Number of sample points.
        resolution : int
            Unit cell voxel resolution.
        E : float
            Young's modulus.
        nu : float
            Poisson's ratio.
        device : str
        dtype : torch.dtype
        """
        if tpms_types is None:
            tpms_types = list(TPMS_REGISTRY.keys())
        self.tpms_types = tpms_types
        self.n_types = len(tpms_types)
        self.vf_min = vf_min
        self.vf_max = vf_max
        self.n_samples = n_samples
        self.resolution = resolution
        self.E = E
        self.nu = nu
        self.device = device
        self.dtype = dtype

        # Precomputed data: {tpms_name: (vf_array, C_eff_array)}
        # vf_array: (n_samples,)
        # C_eff_array: (n_samples, 6, 6)
        self.table = {}
        self.vf_array = torch.linspace(vf_min, vf_max, n_samples,
                                       device=device, dtype=dtype)
        self._computed = False

    def precompute(self, verbose=True):
        """Precompute C_eff for all (type, vf) combinations."""
        homogenizer = NumericalHomogenizer(
            E=self.E, nu=self.nu, device=self.device, dtype=self.dtype
        )

        for t_idx, tpms_name in enumerate(self.tpms_types):
            if verbose:
                print(f"Precomputing homogenization for {tpms_name} "
                      f"({t_idx + 1}/{self.n_types})...")

            C_array = torch.zeros(self.n_samples, 6, 6,
                                  device=self.device, dtype=self.dtype)

            for i, vf in enumerate(self.vf_array):
                vf_val = vf.item()
                C_eff = homogenizer.homogenize_tpms(
                    tpms_name, vf_val, self.resolution
                )
                C_array[i] = C_eff

                if verbose:
                    print(f"  vf={vf_val:.3f}, C11={C_eff[0, 0].item():.4e}")

            self.table[tpms_name] = C_array

        self._computed = True

    def interpolate(self, tpms_name, volume_fraction):
        """Differentiable linear interpolation of C_eff for a given volume fraction.

        Parameters
        ----------
        tpms_name : str
            TPMS type name.
        volume_fraction : torch.Tensor
            Volume fraction value(s). Can be scalar or tensor.

        Returns
        -------
        C_eff : torch.Tensor, shape (..., 6, 6)
        """
        if not self._computed:
            raise RuntimeError("Lookup table not computed. Call precompute() first.")

        C_array = self.table[tpms_name]  # (n_samples, 6, 6)
        vf = volume_fraction

        # Clamp to valid range
        vf_clamped = vf.clamp(self.vf_min, self.vf_max)

        # Find interval index
        dvf = (self.vf_max - self.vf_min) / (self.n_samples - 1)
        idx_float = (vf_clamped - self.vf_min) / dvf
        idx_lo = idx_float.floor().long().clamp(0, self.n_samples - 2)
        t = idx_float - idx_lo.float()  # interpolation parameter in [0, 1]

        # Linear interpolation: C = C[lo] + t * (C[lo+1] - C[lo])
        C_lo = C_array[idx_lo]  # (..., 6, 6)
        C_hi = C_array[idx_lo + 1]  # (..., 6, 6)

        # Expand t for broadcasting
        while t.dim() < C_lo.dim():
            t = t.unsqueeze(-1)

        C_interp = C_lo + t * (C_hi - C_lo)
        return C_interp

    def batch_interpolate(self, volume_fraction):
        """Interpolate C_eff for all TPMS types at given volume fractions.

        Parameters
        ----------
        volume_fraction : torch.Tensor, shape (1, nx, ny, nz)
            Volume fraction field.

        Returns
        -------
        C_per_type : torch.Tensor, shape (n_types, 6, 6, nx, ny, nz)
        """
        vf = volume_fraction.squeeze(0)  # (nx, ny, nz)
        nx, ny, nz = vf.shape

        results = []
        for tpms_name in self.tpms_types:
            vf_flat = vf.reshape(-1)  # (N,)
            C_flat = self.interpolate(tpms_name, vf_flat)  # (N, 6, 6)
            C_spatial = C_flat.reshape(nx, ny, nz, 6, 6)
            C_spatial = C_spatial.permute(3, 4, 0, 1, 2)  # (6, 6, nx, ny, nz)
            results.append(C_spatial)

        return torch.stack(results, dim=0)  # (n_types, 6, 6, nx, ny, nz)

    def voigt_to_9x9(self, C_voigt):
        """Convert 6x6 Voigt stiffness to 9x9 tensor form used by FDM solver.

        Parameters
        ----------
        C_voigt : torch.Tensor, shape (..., 6, 6)

        Returns
        -------
        C_9x9 : torch.Tensor, shape (..., 9, 9)
        """
        # Mapping from 9-idx (i*3+j) to Voigt idx
        # 9: [00,01,02, 10,11,12, 20,21,22]
        # Voigt: [00=0, 11=1, 22=2, 12=3, 02=4, 01=5]
        nine_to_voigt = [0, 5, 4, 5, 1, 3, 4, 3, 2]

        batch_shape = C_voigt.shape[:-2]
        C_9 = torch.zeros(*batch_shape, 9, 9,
                          device=C_voigt.device, dtype=C_voigt.dtype)

        for i9 in range(9):
            for j9 in range(9):
                iv = nine_to_voigt[i9]
                jv = nine_to_voigt[j9]
                C_9[..., i9, j9] = C_voigt[..., iv, jv]

        return C_9

    def batch_interpolate_9x9(self, volume_fraction):
        """Like batch_interpolate but returns 9x9 tensors for FDM solver.

        Parameters
        ----------
        volume_fraction : torch.Tensor, shape (1, nx, ny, nz)

        Returns
        -------
        C_per_type : torch.Tensor, shape (n_types, 9, 9, nx, ny, nz)
        """
        C_voigt = self.batch_interpolate(volume_fraction)  # (T, 6, 6, nx, ny, nz)
        T = C_voigt.shape[0]
        nx, ny, nz = C_voigt.shape[3:]

        # Reshape for voigt_to_9x9: (T*nx*ny*nz, 6, 6) -> (T*nx*ny*nz, 9, 9)
        C_flat = C_voigt.permute(0, 2, 3, 4, 1, 5).contiguous()  # (T, nx, ny, nz, 6, 6) -- wrong
        # Actually let's do it properly
        C_flat = C_voigt.permute(0, 2, 3, 4, 1, 5).contiguous()  # wrong dims
        # C_voigt is (T, 6, 6, nx, ny, nz)
        # We want (T, nx, ny, nz, 6, 6)
        C_reshaped = C_voigt.permute(0, 3, 4, 5, 1, 2).contiguous()  # (T, nx, ny, nz, 6, 6)
        C_9 = self.voigt_to_9x9(C_reshaped)  # (T, nx, ny, nz, 9, 9)
        C_9 = C_9.permute(0, 4, 5, 1, 2, 3).contiguous()  # (T, 9, 9, nx, ny, nz)

        return C_9

    def save(self, path):
        """Save lookup table to disk."""
        data = {
            'tpms_types': self.tpms_types,
            'vf_array': self.vf_array.cpu(),
            'n_samples': self.n_samples,
            'vf_min': self.vf_min,
            'vf_max': self.vf_max,
            'resolution': self.resolution,
            'E': self.E,
            'nu': self.nu,
            'table': {k: v.cpu() for k, v in self.table.items()},
        }
        torch.save(data, path)

    def load(self, path):
        """Load lookup table from disk."""
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.tpms_types = data['tpms_types']
        self.n_types = len(self.tpms_types)
        self.vf_array = data['vf_array'].to(device=self.device, dtype=self.dtype)
        self.n_samples = data['n_samples']
        self.vf_min = data['vf_min']
        self.vf_max = data['vf_max']
        self.resolution = data['resolution']
        self.E = data['E']
        self.nu = data['nu']
        self.table = {k: v.to(device=self.device, dtype=self.dtype)
                      for k, v in data['table'].items()}
        self._computed = True
