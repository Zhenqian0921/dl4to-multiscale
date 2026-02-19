"""
Multiscale anisotropic FDM PDE solver.

Extends the existing FDM solver to support per-voxel anisotropic stiffness
tensors (C_field) instead of scalar SIMP density interpolation.

The key modification: instead of `_apply_theta_p(sigma, theta, p) + _G(eps)`,
we use `_apply_C_eff(eps, C_field)` where C_field is a (9, 9, nx, ny, nz)
tensor representing the local constitutive relation at each voxel.
"""

__all__ = ['MultiscaleFDM']

import torch
import numpy as np
from scipy.sparse.linalg import factorized, spsolve
from scipy.sparse import csc_matrix, diags

from .pde import (
    PDESolver, SparseLinearSolver, AutogradLinearSolver,
    FDMDerivatives, FDMAdjointDerivatives, FDMAssembly,
)
from .utils import get_σ_vm


class MultiscaleFDM(PDESolver):
    """FDM solver using per-voxel anisotropic stiffness tensors.

    Instead of isotropic SIMP interpolation (theta^p * G), this solver uses
    a full 9x9 constitutive matrix at each voxel: sigma_i(x) = C_ij(x) * eps_j(x).

    The system operator is: A(u) = J^T * C_eff * J * u

    Reuses the existing FDM derivative operators (_J, _J_adj) and the
    AutogradLinearSolver for differentiable backward pass.
    """

    def __init__(self,
                 theta_min=1e-6,
                 use_forward_differences=True,
                 assemble_tensors_when_passed_to_problem=True):
        """
        Parameters
        ----------
        theta_min : float
            Minimum stiffness scaling to avoid singular matrices.
        use_forward_differences : bool
            Whether to use forward (vs central) finite differences.
        assemble_tensors_when_passed_to_problem : bool
            Whether to pre-assemble tensors when problem is assigned.
        """
        self._theta_min = theta_min
        self._linear_solver = SparseLinearSolver(use_umfpack=True, factorize=True)
        self.use_forward_differences = use_forward_differences
        self.assemble_tensors_when_passed_to_problem = assemble_tensors_when_passed_to_problem
        self.assembled_tensors = False
        self._C_field = None
        super().__init__(assemble_tensors_when_passed_to_problem)

    @property
    def problem(self):
        return self._problem

    @property
    def shape(self):
        return self.problem.shape

    @property
    def Omega_dirichlet(self):
        return self.problem.Ω_dirichlet

    @property
    def theta_min(self):
        return self._theta_min

    @property
    def h(self):
        return self.problem.h

    @property
    def linear_solver(self):
        return self._linear_solver

    def set_stiffness_field(self, C_field):
        """Set the per-voxel stiffness field.

        Parameters
        ----------
        C_field : torch.Tensor, shape (9, 9, nx, ny, nz)
            Per-voxel constitutive matrix in 9x9 tensor form.
            C_field[i, j, x, y, z] is the (i,j) entry of C at voxel (x,y,z).
        """
        self._C_field = C_field

    def assemble_tensors(self, problem):
        """Pre-assemble geometry-dependent tensors.

        Parameters
        ----------
        problem : dl4to.problem.Problem
        """
        self._problem = problem.clone()
        self._Omega_dirichlet_diags = diags(
            self.Omega_dirichlet.flatten().int().numpy()
        )

        # Assemble J matrix (strain operator) with Dirichlet BCs
        self._J_mat = FDMAssembly.assemble_operator(
            operator=self._J, shape=self.shape,
            Ω_dirichlet=self.Omega_dirichlet,
            filter_shape=3
        )
        self._Jt_mat = self._J_mat.transpose()

        self._b = self._get_b()
        self.assembled_tensors = True

    def _J(self, u, dirichlet=False):
        """Strain operator: u (3, nx, ny, nz) -> eps (9, nx, ny, nz)."""
        J = lambda u: torch.cat([
            FDMDerivatives.du_dx(u, self.h, self.use_forward_differences),
            FDMDerivatives.du_dy(u, self.h, self.use_forward_differences),
            FDMDerivatives.du_dz(u, self.h, self.use_forward_differences)
        ], dim=0)

        if dirichlet:
            return FDMAssembly.apply_dirichlet_zero_columns_to_operator(
                J, self.Omega_dirichlet
            )(u)
        return J(u)

    def _J_adj(self, sigma, dirichlet=False):
        """Adjoint strain operator: sigma (9, nx, ny, nz) -> u (3, nx, ny, nz)."""
        Jt = lambda sigma: (
            FDMAdjointDerivatives.du_dx_adj(sigma[:3], self.h, self.use_forward_differences)
            + FDMAdjointDerivatives.du_dy_adj(sigma[3:6], self.h, self.use_forward_differences)
            + FDMAdjointDerivatives.du_dz_adj(sigma[6:], self.h, self.use_forward_differences)
        )

        if dirichlet:
            return FDMAssembly.apply_dirichlet_zero_rows_to_operator(
                Jt, self.Omega_dirichlet
            )(sigma)
        return Jt(sigma)

    def _apply_C_eff(self, epsilon, C_field):
        """Apply per-voxel anisotropic constitutive relation.

        sigma_i(x) = C_ij(x) * eps_j(x)

        Parameters
        ----------
        epsilon : torch.Tensor, shape (9, nx, ny, nz)
            Strain field.
        C_field : torch.Tensor, shape (9, 9, nx, ny, nz)
            Per-voxel stiffness tensor.

        Returns
        -------
        sigma : torch.Tensor, shape (9, nx, ny, nz)
        """
        return torch.einsum('ijxyz, jxyz -> ixyz', C_field, epsilon)

    def _A(self, u, C_field, dirichlet=True):
        """System operator: y = J^T * C_eff * J * u.

        Parameters
        ----------
        u : torch.Tensor, shape (3, nx, ny, nz) or flattened
        C_field : torch.Tensor, shape (9, 9, nx, ny, nz)
        dirichlet : bool

        Returns
        -------
        y : torch.Tensor, same shape as u
        """
        nx, ny, nz = self.shape
        u = u.view(3, nx, ny, nz)

        epsilon = self._J(u, dirichlet)
        sigma = self._apply_C_eff(epsilon, C_field)
        y = self._J_adj(sigma, dirichlet)

        if dirichlet:
            y[self.Omega_dirichlet] = u.clone()[self.Omega_dirichlet]
        return y

    def _assemble_C_block_diag(self, C_field):
        """Assemble C_field as a block-diagonal sparse matrix.

        The C_field (9, 9, nx, ny, nz) becomes a sparse matrix of size
        (9*N, 9*N) where N = nx*ny*nz.
        """
        C = C_field.detach().cpu()
        nx, ny, nz = C.shape[2:]
        N = nx * ny * nz

        rows = []
        cols = []
        vals = []

        C_flat = C.reshape(9, 9, N)

        for i in range(9):
            for j in range(9):
                c_vals = C_flat[i, j, :].numpy()
                nz_mask = np.abs(c_vals) > 1e-15
                indices = np.where(nz_mask)[0]

                if len(indices) > 0:
                    row_idx = i * N + indices
                    col_idx = j * N + indices
                    rows.extend(row_idx.tolist())
                    cols.extend(col_idx.tolist())
                    vals.extend(c_vals[nz_mask].tolist())

        C_sparse = csc_matrix((vals, (rows, cols)), shape=(9 * N, 9 * N))
        return C_sparse

    def _assemble_A(self, C_field):
        """Assemble system matrix: A = J^T * BlockDiag(C_eff) * J + Dirichlet.

        Parameters
        ----------
        C_field : torch.Tensor, shape (9, 9, nx, ny, nz)

        Returns
        -------
        A : scipy.sparse.csc_matrix
        """
        # Assemble J matrix if not done
        if not self.assembled_tensors:
            raise RuntimeError("Call assemble_tensors() first.")

        # Build block-diagonal C matrix
        C_diag = self._assemble_C_block_diag(C_field)

        # A = Jt * C * J + Dirichlet identity
        GJ = C_diag.dot(self._J_mat)
        A = self._Jt_mat.dot(GJ) + self._Omega_dirichlet_diags

        return csc_matrix(A)

    def _get_b(self):
        """Get the right-hand side vector (normalized forces)."""
        b = self.problem.F.clone()
        b[self.Omega_dirichlet] = 0
        # Note: for multiscale, we don't divide by E since C_field already
        # contains absolute stiffness values. However, if C_field is normalized
        # (e.g., from lookup table with E=1), we still normalize.
        b /= self.problem.E
        return b

    def _get_u(self, solution, C_field=None, binary=False):
        """Solve for displacement field.

        Parameters
        ----------
        solution : Solution
        C_field : torch.Tensor, shape (9, 9, nx, ny, nz), optional
            If None, uses self._C_field.
        binary : bool

        Returns
        -------
        u : torch.Tensor, shape (3, nx, ny, nz)
        """
        if C_field is None:
            C_field = self._C_field
        if C_field is None:
            raise ValueError("No stiffness field set. Call set_stiffness_field() first.")

        if not self.assembled_tensors:
            self.assemble_tensors(solution.problem)

        # We need theta for the autograd solver's backward pass
        theta = solution.get_θ(binary).clone().clamp(self.theta_min, 1)

        # Build operator and matrix
        A_op = lambda u, theta: self._A(u, C_field)
        A_mat = self._assemble_A(C_field.cpu())

        u = self._linear_solver(theta.cpu(), A_op, self._b.flatten(), A_mat)
        u = u.view(3, *self.shape).to(C_field.device)

        return u

    def _get_sigma(self, u, C_field):
        """Compute stress from displacement.

        Parameters
        ----------
        u : torch.Tensor, shape (3, nx, ny, nz)
        C_field : torch.Tensor, shape (9, 9, nx, ny, nz)

        Returns
        -------
        sigma : torch.Tensor, shape (9, nx, ny, nz)
        """
        eps = self._J(u)
        sigma = self._apply_C_eff(eps, C_field)
        # Scale by E (denormalize) if C_field was normalized
        sigma = sigma * self.problem.E
        return sigma

    def solve_pde(self, solution, p=1.0, binary=False, C_field=None):
        """Solve the PDE with per-voxel anisotropic stiffness.

        Parameters
        ----------
        solution : Solution
        p : float
            Unused (kept for API compatibility).
        binary : bool
        C_field : torch.Tensor, optional
            If None, uses self._C_field.

        Returns
        -------
        u : torch.Tensor, shape (3, nx, ny, nz)
        sigma : torch.Tensor, shape (9, nx, ny, nz)
        sigma_vm : torch.Tensor, shape (1, nx, ny, nz)
        """
        if C_field is not None:
            self.set_stiffness_field(C_field)

        u = self._get_u(solution, self._C_field, binary)
        sigma = self._get_sigma(u, self._C_field)
        sigma_vm = get_σ_vm(sigma)

        return u, sigma, sigma_vm

    def solve_pde_with_stiffness(self, solution, C_field, binary=False):
        """Convenience method: solve PDE with explicit stiffness field.

        Parameters
        ----------
        solution : Solution
        C_field : torch.Tensor, shape (9, 9, nx, ny, nz)
        binary : bool

        Returns
        -------
        u, sigma, sigma_vm
        """
        self.set_stiffness_field(C_field)
        return self.solve_pde(solution, binary=binary)
