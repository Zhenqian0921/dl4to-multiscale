"""
Multiscale TPMS topology optimization solver.

Implements MultiscaleSIMPIterator and MultiscaleSIMP, following the same
patterns as SIMPIterator and SIMP from topo_solvers.py.

The key difference: instead of optimizing a scalar density field via SIMP,
we optimize TPMS parameters (volume fraction + type weights) that map to
per-voxel anisotropic stiffness tensors through a homogenization lookup table.
"""

__all__ = ['MultiscaleSIMPIterator', 'MultiscaleSIMP']

import time
import torch
from collections import defaultdict
from tqdm import tqdm

from .solution import Solution
from .criteria import VolumeFraction, Binariness
from .topo_solvers import TopoSolver
from .multiscale_pde import MultiscaleFDM
from .multiscale_representer import MultiscaleRepresenter


class MultiscaleSIMPIterator:
    """Performs one iteration of multiscale TPMS optimization.

    Analogous to SIMPIterator but uses MultiscaleRepresenter and MultiscaleFDM.
    """

    def __init__(self, problem, criterion, representer, pde_solver, lr=3e-2,
                 temperature_decay=1.0):
        """
        Parameters
        ----------
        problem : dl4to.problem.Problem
        criterion : dl4to.criteria.Criterion
        representer : MultiscaleRepresenter
        pde_solver : MultiscaleFDM
        lr : float
            Learning rate for Adam optimizer.
        temperature_decay : float
            Factor to multiply temperature each iteration (< 1 sharpens type selection).
        """
        self.problem = problem
        self.criterion = criterion
        self.representer = representer
        self.pde_solver = pde_solver
        self.lr = lr
        self.temperature_decay = temperature_decay
        self.logs = defaultdict(list)

        self.volume_crit = VolumeFraction()
        self.optimizer = torch.optim.Adam(self.representer.parameters(), lr=self.lr)

    def _extend_logs(self, solution, loss, volume, tick, sigma_vm):
        self.logs["losses"].append(loss.item())
        self.logs["volumes"].append(volume.item())
        self.logs["durations"].append(time.time() - tick)
        if sigma_vm is not None:
            self.logs["relative_max_sigma_vm"].append(
                sigma_vm.max().item() / self.problem.σ_ys
            )

    def __call__(self):
        """Run one optimization iteration.

        Returns
        -------
        solution : Solution
            Solution with updated density (volume fraction).
        """
        tick = time.time()

        # Forward pass: get stiffness field and volume fraction
        C_field, vf = self.representer()

        # Create solution with vf as density
        solution = Solution(
            problem=self.problem,
            θ=vf.clamp(0, 1),
            enforce_θ_on_Ω_design=True,
        )

        # Solve PDE with anisotropic stiffness
        self.pde_solver.set_stiffness_field(C_field)
        u, sigma, sigma_vm = self.pde_solver.solve_pde(solution)

        # Compute loss and backpropagate
        loss = self.criterion([solution])
        volume = self.volume_crit([solution])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Anneal temperature for type selection
        if self.temperature_decay != 1.0:
            self.representer.set_temperature(
                self.representer.temperature * self.temperature_decay
            )

        # Update solution with post-optimization params
        with torch.no_grad():
            C_field_new, vf_new = self.representer()
            solution.θ = vf_new.clamp(0, 1)
            solution.tpms_params = self.representer.get_optimized_params()

        self._extend_logs(solution, loss, volume, tick, sigma_vm)
        solution.logs = self.logs

        return solution


class MultiscaleSIMP(TopoSolver):
    """Multiscale TPMS topology optimization solver.

    Follows the SIMP pattern but optimizes TPMS lattice parameters
    instead of scalar density. Each voxel is assigned a TPMS type
    and volume fraction, which map to anisotropic effective stiffness
    via a precomputed homogenization lookup table.
    """

    def __init__(self, criterion, homogenization_table,
                 n_iterations=100, verbose=True, lr=3e-2,
                 initial_vf=0.3, vf_min=0.05, vf_max=0.95,
                 filter_size=3, use_filter=True,
                 temperature_init=1.0, temperature_decay=0.99,
                 return_intermediate_solutions=False):
        """
        Parameters
        ----------
        criterion : dl4to.criteria.Criterion
            Objective function (e.g., Compliance + VolumeConstraint).
        homogenization_table : HomogenizationLookupTable
            Precomputed C_eff lookup table.
        n_iterations : int
            Number of optimization iterations.
        verbose : bool
        lr : float
            Learning rate.
        initial_vf : float
            Initial volume fraction.
        vf_min, vf_max : float
            Volume fraction bounds.
        filter_size : int
            Spatial smoothing filter size.
        use_filter : bool
        temperature_init : float
            Initial softmax temperature.
        temperature_decay : float
            Temperature decay factor per iteration.
        return_intermediate_solutions : bool
        """
        super().__init__(device="cpu", name="MultiscaleSIMP")

        self.criterion = criterion
        self.table = homogenization_table
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.lr = lr
        self.initial_vf = initial_vf
        self.vf_min = vf_min
        self.vf_max = vf_max
        self.filter_size = filter_size
        self.use_filter = use_filter
        self.temperature_init = temperature_init
        self.temperature_decay = temperature_decay
        self.return_intermediate_solutions = return_intermediate_solutions

    def _run_iterations(self, iterator):
        """Run the optimization loop.

        Parameters
        ----------
        iterator : MultiscaleSIMPIterator

        Returns
        -------
        solution or list of solutions
        """
        solutions = []
        iters = range(self.n_iterations)
        if self.verbose:
            iters = tqdm(iters, desc="MultiscaleSIMP")

        for i in iters:
            solution = iterator()

            if self.verbose and hasattr(iters, 'set_postfix'):
                logs = solution.logs
                postfix = {}
                if logs["losses"]:
                    postfix["loss"] = f"{logs['losses'][-1]:.4e}"
                if logs["volumes"]:
                    postfix["vf"] = f"{logs['volumes'][-1]:.3f}"
                iters.set_postfix(postfix)

            if self.return_intermediate_solutions:
                solutions.append(solution)

        if self.return_intermediate_solutions:
            return solutions
        return solution

    def _get_new_solution(self, solution):
        """Create and run a multiscale optimization for one problem.

        Parameters
        ----------
        solution : Solution
            Initial solution (problem is extracted from it).

        Returns
        -------
        solution : Solution
            Optimized solution with tpms_params attribute.
        """
        problem = solution.problem

        # Create representer
        representer = MultiscaleRepresenter(
            problem=problem,
            homogenization_table=self.table,
            initial_vf=self.initial_vf,
            vf_min=self.vf_min,
            vf_max=self.vf_max,
            filter_size=self.filter_size,
            use_filter=self.use_filter,
            temperature=self.temperature_init,
        )

        # Create PDE solver
        pde_solver = MultiscaleFDM()
        pde_solver.assemble_tensors(problem)

        # Also attach the pde_solver to the problem for criterion evaluation
        problem._pde_solver = pde_solver

        # Create iterator
        iterator = MultiscaleSIMPIterator(
            problem=problem,
            criterion=self.criterion,
            representer=representer,
            pde_solver=pde_solver,
            lr=self.lr,
            temperature_decay=self.temperature_decay,
        )

        # Run optimization
        result = self._run_iterations(iterator)
        return result

    def _get_new_solutions(self, solutions, eval_mode=False):
        """Run multiscale optimization for each solution/problem.

        Parameters
        ----------
        solutions : list of Solution
        eval_mode : bool (unused for SIMP-like optimization)

        Returns
        -------
        list of Solution
        """
        results = []
        for solution in solutions:
            result = self._get_new_solution(solution)
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)
        return results
