"""
Multiscale-specific optimization criteria.

Provides additional constraints for multiscale TPMS optimization:
- GradedPropertyConstraint: penalizes abrupt changes in TPMS parameters
- LocalVolumeConstraint: enforces local volume fraction bounds
"""

__all__ = ['GradedPropertyConstraint', 'LocalVolumeConstraint']

import torch
from torch.nn.functional import softplus, relu

from .criteria import UnsupervisedCriterion


class GradedPropertyConstraint(UnsupervisedCriterion):
    """Penalizes abrupt spatial changes in TPMS parameters for manufacturability.

    Computes the spatial gradient magnitude of volume fraction and TPMS type
    weights, and penalizes values exceeding a threshold. This promotes smooth
    gradation of lattice properties across the design domain.
    """

    def __init__(self, max_gradient=0.3, weight_vf=1.0, weight_type=0.5,
                 threshold_fct='softplus'):
        """
        Parameters
        ----------
        max_gradient : float
            Maximum allowed gradient magnitude before penalty kicks in.
        weight_vf : float
            Weight for volume fraction gradient penalty.
        weight_type : float
            Weight for TPMS type weight gradient penalty.
        threshold_fct : str
            'softplus' or 'relu'.
        """
        super().__init__(
            name='graded_property_constraint',
            compute_only_on_design_space=True
        )
        self.max_gradient = max_gradient
        self.weight_vf = weight_vf
        self.weight_type = weight_type

        if threshold_fct == 'softplus':
            self.threshold_fct = softplus
        elif threshold_fct == 'relu':
            self.threshold_fct = relu
        else:
            raise ValueError("`threshold_fct` must be one of ['softplus', 'relu'].")

    def _spatial_gradient_magnitude(self, field):
        """Compute spatial gradient magnitude using finite differences.

        Parameters
        ----------
        field : torch.Tensor, shape (C, nx, ny, nz) or (1, nx, ny, nz)

        Returns
        -------
        grad_mag : torch.Tensor, shape (C, nx, ny, nz)
        """
        # Forward differences along each axis
        dx = torch.zeros_like(field)
        dy = torch.zeros_like(field)
        dz = torch.zeros_like(field)

        dx[:, :-1, :, :] = field[:, 1:, :, :] - field[:, :-1, :, :]
        dy[:, :, :-1, :] = field[:, :, 1:, :] - field[:, :, :-1, :]
        dz[:, :, :, :-1] = field[:, :, :, 1:] - field[:, :, :, :-1]

        grad_mag = (dx ** 2 + dy ** 2 + dz ** 2).sqrt()
        return grad_mag

    def __call__(self, solutions, gt_solutions=None, binary=False):
        """Evaluate graded property constraint.

        Expects solutions to have `tpms_params` attribute with 'vf' and
        'type_weights' fields. Falls back to computing gradient on theta.

        Parameters
        ----------
        solutions : list of Solution
        gt_solutions : ignored
        binary : bool

        Returns
        -------
        penalty : torch.Tensor, shape (n_solutions,)
        """
        solutions = self._convert_to_list(solutions)
        penalties = []

        for solution in solutions:
            penalty = torch.tensor(0.0, device=solution.θ.device)

            if hasattr(solution, 'tpms_params') and solution.tpms_params is not None:
                params = solution.tpms_params

                # Volume fraction gradient penalty
                if 'vf' in params and self.weight_vf > 0:
                    vf = params['vf']
                    grad_vf = self._spatial_gradient_magnitude(vf)
                    excess = grad_vf - self.max_gradient
                    vf_penalty = self.threshold_fct(excess).mean()
                    penalty = penalty + self.weight_vf * vf_penalty

                # Type weights gradient penalty
                if 'type_weights' in params and self.weight_type > 0:
                    tw = params['type_weights']
                    grad_tw = self._spatial_gradient_magnitude(tw)
                    excess = grad_tw - self.max_gradient
                    tw_penalty = self.threshold_fct(excess).mean()
                    penalty = penalty + self.weight_type * tw_penalty
            else:
                # Fallback: compute gradient on density theta
                theta = solution.get_θ(binary=binary)
                grad_theta = self._spatial_gradient_magnitude(theta)
                excess = grad_theta - self.max_gradient
                penalty = self.threshold_fct(excess).mean()

            penalties.append(penalty)

        return torch.stack(penalties)


class LocalVolumeConstraint(UnsupervisedCriterion):
    """Constraint on local (neighborhood) volume fraction.

    Uses average pooling to compute local volume fraction in a neighborhood,
    then penalizes values above a maximum threshold.
    """

    def __init__(self, max_local_vf=0.5, neighborhood_size=3,
                 threshold_fct='softplus'):
        """
        Parameters
        ----------
        max_local_vf : float
            Maximum allowed local volume fraction.
        neighborhood_size : int
            Size of the averaging neighborhood (cubic).
        threshold_fct : str
            'softplus' or 'relu'.
        """
        super().__init__(
            name='local_volume_constraint',
            compute_only_on_design_space=False
        )
        self.max_local_vf = max_local_vf
        self.neighborhood_size = neighborhood_size

        if threshold_fct == 'softplus':
            self.threshold_fct = softplus
        elif threshold_fct == 'relu':
            self.threshold_fct = relu
        else:
            raise ValueError("`threshold_fct` must be one of ['softplus', 'relu'].")

    def __call__(self, solutions, gt_solutions=None, binary=False):
        """Evaluate local volume constraint.

        Parameters
        ----------
        solutions : list of Solution
        gt_solutions : ignored
        binary : bool

        Returns
        -------
        penalty : torch.Tensor, shape (n_solutions,)
        """
        solutions = self._convert_to_list(solutions)
        penalties = []
        k = self.neighborhood_size
        pad = k // 2

        for solution in solutions:
            theta = solution.get_θ(binary=binary)  # (1, nx, ny, nz)

            # Compute local average via 3D average pooling
            # Need (batch, channel, D, H, W) format
            theta_5d = theta.unsqueeze(0)  # (1, 1, nx, ny, nz)
            local_avg = torch.nn.functional.avg_pool3d(
                theta_5d, kernel_size=k, stride=1, padding=pad
            ).squeeze(0)  # (1, nx, ny, nz)

            # Handle size mismatch from pooling
            if local_avg.shape != theta.shape:
                local_avg = local_avg[:, :theta.shape[1], :theta.shape[2], :theta.shape[3]]

            excess = local_avg - self.max_local_vf
            penalty = self.threshold_fct(excess).mean()
            penalties.append(penalty)

        return torch.stack(penalties)
