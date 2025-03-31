"""
Abstract base class for Data-Driven Predictive Control (DPC).

This class provides a structured interface for all DPC-based controllers, ensuring:

- Dynamic subclass registration for flexible controller instantiation.
- Common functionality such as Hankel matrix construction and constraint handling.
- Standardized optimization problem formulation.

Subclasses must implement:

- build_optimization_problem()
- calculate_offline_data()
- compute_closed_form_gains()
"""

import logging
from abc import ABC, abstractmethod
from cmath import sqrt
from typing import Type, Callable, Dict, Optional

import cvxpy as cp
import numpy as np
from cvxpy import Constraint

from do_ddpc.control_utils.control_structs import HankelMatrices, InputOutputTrajectory, Bounds
from do_ddpc.ddpc.ddpc_structs import (
    DPCParameters,
    DPCClosedFormSolutionMatrices,
    DPCPredictorMatrices,
    DPCRegularizationMatrices,
)
from do_ddpc.ddpc.ddpc_utils import DPCUtils

logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes
class DPC(ABC):
    """
    Abstract base class for Data-Driven Predictive Control (DPC).

    This serves as the base class for all DPC-based controllers, enforcing a common API
    and providing essential functionality such as updating measurements and reference trajectory,
    solving the optimization problem, returning the next optimal control action.

    Attributes:
        ddpc_params (DPCParameters): Controller configuration parameters for the DPC.
        dims (Dimensions): Calculated dimensions based on controller parameters.
        hankel_matrices (HankelMatrices): Constructed Hankel matrices from trajectory data.
        pred_matrices (DPCPredictorMatrices): Prediction matrices for predicting `y_f`.
        reg_matrices (DPCRegularizationMatrices): Regularization matrices to calculate cost.
        constr (list[Constraint]): List of constraints for the CVXPY optimization problem.
        cost (cvxpy.Expression): Total cost function for the optimization problem.
        problem (cvxpy.Problem): CVXPY problem instance for solving the optimization.
        valid_optimization_problem (bool): Flag indicating if the optimization problem is valid.
        u_f_cp (cp.Variable): Control input decision variable for the optimization.
        y_f_cp (cp.Variable): Output variable decision for the optimization.
        y_r_cp (cp.Parameter): Reference output parameter for the optimization problem.
        u_r_cp (cp.Parameter): Reference control input parameter for the optimization problem.
        z_p_cp (cp.Parameter): Parameter related to perturbations or uncertainties in the system.
        u_f (np.ndarray): Offline computed control input (closed-form solution).
        is_unconstrained (bool): Flag indicating if the optimization is unconstrained.
        cf_gains (np.ndarray): Closed-form gains computed for the controller.
        control_step (int): Counter for the number of control steps taken.
        use_mpc_cf (bool): Flag indicating if MPC closed-form is used.

    Args:
        ddpc_params (DPCParameters): Controller configuration parameters.
        training_data (InputOutputTrajectory): Collected trajectory data.
        **Kwargs: Additional keyword arguments.
    """

    _registry: Dict[str, Type["DPC"]] = {}  # Stores registered subclasses for dynamic instantiation

    @classmethod
    def register(cls, ddpc_type: str) -> Callable[[Type["DPC"]], Type["DPC"]]:
        """
        Decorator to register subclasses for dynamic instantiation.

        Usage:
            @DPC.register("my_ddpc")
            class MyDPC(DPC):
                ...

        Args:
            ddpc_type (str): Name of the DPC subclass type.

        Returns:
            Callable[[Type["DPC"]], Type["DPC"]]: The decorator function.
        """

        def decorator(subclass: Type["DPC"]) -> Type["DPC"]:
            cls._registry[ddpc_type.lower()] = subclass
            return subclass

        return decorator

    @staticmethod
    def instantiate(ddpc_type: str, *args, **kwargs):
        """
        Factory method to create an instance of a registered DPC subclass.

        Args:
            ddpc_type (str): The type of DPC controller to instantiate.
            *args: Positional arguments for the subclass constructor.
            **kwargs: Keyword arguments for the subclass constructor.

        Returns:
            DPC: An instance of the corresponding DPC subclass.

        Raises:
            ValueError: If the specified `ddpc_type` is not registered.
        """
        subclass = DPC._registry.get(ddpc_type.lower())
        if subclass:
            return subclass(*args, **kwargs)
        raise ValueError(f"Unknown DPC type: {ddpc_type}")

    @abstractmethod
    def calculate_predictor_matrices(self) -> DPCPredictorMatrices:
        """Calculates predictor matrices for the DPC controller."""

    @abstractmethod
    def calculate_regularization_matrices(self) -> DPCRegularizationMatrices:
        """Calculates regularization matrices for optimal control."""

    @abstractmethod
    def calculate_closed_form_solution_matrices(self) -> Optional[DPCClosedFormSolutionMatrices]:
        """
        Abstract method to be implemented by subclasses to calculate closed-form gains.

        If the implementation does not support closed-form solutions, this method returns None.
        Otherwise, it validates the computed gains and sets `self.implemented_cf` to True.

        Returns:
            Optional[DPCClosedFormSolutionMatrices]: The computed gain matrices if available, otherwise None.
        """

    @abstractmethod
    def get_regularization_cost_expression(self) -> cp.Expression:
        """
        Calculated and returns the regularization cost expression r(.)

         Returns:
            cp.Expression: The CVXPY expression for the regularization cost.
        """

    @abstractmethod
    def get_predictor_constraint_expression(self) -> cp.constraints.Constraint:
        """
        Calculates and returns the CVXPY expression for the predictor constraint f.

        Returns:
            cp.constraints.Constraint: The CVXPY constraint for the predictor constraint.
        """

    # pylint: disable=unused-argument
    def __init__(self, ddpc_params: DPCParameters, training_data: InputOutputTrajectory, **Kwargs):
        """
        Initializes the DPC controller.

        Args:
            ddpc_params (DPCParameters): Controller configuration parameters.
            training_data (InputOutputTrajectory): Collected trajectory data.
            **Kwargs: Additional keyword arguments.
        """

        # Validate arguments
        m, p, n_samples = DPCUtils.check_valid_trajectory_data(training_data)
        DPCUtils.check_valid_controller_parameters(ddpc_params, m, p)

        self.ddpc_params = ddpc_params

        # Store parameters

        self.dims = DPCUtils.calculate_dimensions(ddpc_params, m, p)
        self.hankel_matrices = self._construct_hankel_matrices(training_data, n_samples)

        # Compute offline data
        self.pred_matrices = self.calculate_predictor_matrices()
        self.reg_matrices = self.calculate_regularization_matrices()

        self.cf_matrices = self.calculate_closed_form_solution_matrices()
        if self.cf_matrices is not None:
            DPCUtils.check_valid_closed_form_gains(self.dims, self.cf_matrices)

        # Closed-form solution
        self.u_f = np.array((self.dims.n_u_f,))
        self.is_unconstrained = True

        # CVXPY Optimization Problem Initialization
        self.constr: list[Constraint] = []
        self.cost = 0
        self.problem = cp.Problem(cp.Minimize(self.cost), self.constr)  # type: ignore
        self.valid_optimization_problem = False

        # CVXPY Decision Variables
        self.u_f_cp = cp.Variable(self.dims.n_u_f)
        self.y_f_cp = cp.Variable(self.dims.n_y_f)

        # CVXPY Parameters
        self.y_r_cp = cp.Parameter(shape=self.dims.n_y_f, value=np.zeros(self.dims.n_y_f))
        self.u_r_cp = cp.Parameter(shape=self.dims.n_u_f, value=np.zeros(self.dims.n_u_f))
        self.z_p_cp = cp.Parameter(shape=self.dims.n_z_p, value=np.zeros(self.dims.n_z_p))

        # Control step tracker
        self.control_step = 0

        # MPC CF (special case)
        self.use_mpc_cf = False

    def build_optimization_problem(self):
        r"""
        Constructs the DPC optimization problem.

        .. math::

            \min_{u_f, \cdot} &\quad \|y_f - y_r\|_Q^2 + \|u_f - u_r\|_R^2 + r(\cdot) \\
            \text{s.t.} &\quad y_f = f(\cdot) \\
            &\quad u_f \in \mathcal{U}, \quad y_f \in \mathcal{Y}

        where :math:`r(\cdot)`, :math:`f(\cdot)` are linear functions
        which utilize matrices calculated in the **Offline Calculations**.
        Additional slack decision variables can be introduced.
        :math:`r(\cdot)`, :math:`f(\cdot)` differ between different `DPC` algorithm.

        The constraints for :math:`u_f`, :math:`y_f` are handled by other methods.
        """
        # Quadratic cost for being away from the reference.
        self.cost = cp.quad_form(self.y_f_cp - self.y_r_cp, self.ddpc_params.Q_horizon)
        self.cost += cp.quad_form(self.u_f_cp - self.u_r_cp, self.ddpc_params.R_horizon)

        # Regularization cost
        self.cost += self.get_regularization_cost_expression()

        # Add Delta u_f cost
        self.cost += self._calculate_delta_u_f_cost()

        # Multistep predictor constraint
        self.constr.append(self.get_predictor_constraint_expression())

        # Construct problem
        self.problem = cp.Problem(cp.Minimize(self.cost), self.constr)
        self.valid_optimization_problem = True

        logger.debug("Optimization problem has been built.")

    def solve(self, verbose: bool = False, solver: str = cp.SCS, **kwargs):
        """
        Solves the optimization problem to calculate the optimal control input.

        Recommended to use `SCS` as it does not require a license.
        MOSEK should be used for better performance if a license is available.

        Args:
            verbose (bool): If True, solver output is displayed.
            solver (str): Solver to use (default: SCS).
            **kwargs: Additional solver parameters.

        Raises:
            ValueError: If the optimization problem is not properly built.
            ValueError: If the cf_gains are None while using the closed-form solution.
            RuntimeError: If the solver encounters an issue or produces an infeasible result.
            Warning: If another solver than Mosek is used.
        """
        if self._is_closed_form_possible():
            if self.cf_matrices is None:
                raise ValueError("Closed-form gains are unexpectedly None despite being marked as possible.")

            self.u_f = (
                self.cf_matrices.K_z_p @ self.z_p_cp.value
                + self.cf_matrices.K_u_r @ self.u_r_cp.value
                + self.cf_matrices.K_y_r @ self.y_r_cp.value
            )

            self.control_step = 0
            return

        try:
            if not self.valid_optimization_problem:
                raise ValueError(
                    "Optimization problem is not valid. "
                    "Ensure that `build_optimization_problem(self)` has been called before solving."
                )

            self.problem.solve(solver=solver, verbose=verbose, **kwargs)

            if self.problem.status in ["infeasible", "unbounded"]:
                # Verbose output of the solver
                self.problem.solve(solver=solver, verbose=True, **kwargs)

                logger.error(
                    "Optimization failed due to an %s problem. \n"
                    "If the solver status is `DUAL_INFEASIBLE`, "
                    "it likely means the constraints are too restrictive "
                    "or conflict with the system dynamics (i.e. system lag). \n"
                    "Consider relaxing the constraints "
                    "or verifying whether the constraints can be satisfied given the system.",
                    self.problem.status,
                )
                raise RuntimeError(f"Optimization failed: Problem status is {self.problem.status}.")

            if self.problem.status in ["optimal_inaccurate"]:
                logger.warning("Optimization result is inaccurate.")

        except cp.error.SolverError as e:
            raise RuntimeError(f"Solver encountered an error: {str(e)}") from e

        self.control_step = 0

    def add_custom_constraints(self, constraints: list):
        """
        Adds custom constraints to the optimization problem.

        Note:
            - Users should access `cvxpy` parameters and decision variables directly from the DPC class.
            - All such attributes are denoted by the `_cp` suffix.
            - Ensure that added constraints do not conflict with existing ones.
            - For simple linear inequality constraints, prefer using `set_input_constraint` or `set_output_constraint`.

        Args:
            constraints (list[Constraint]): A list of `cvxpy` constraints.
        """
        self.constr += constraints
        self.valid_optimization_problem = False
        self.is_unconstrained = False

        logger.info(
            "Custom constraints have been added to the optimization problem. "
            "Ensure they do not conflict with existing constraints to avoid infeasibility.\n"
            "For standard linear constraints, use `add_linear_constraints()` function instead.\n"
            "Adding custom constraints requires familiarity with the `CVXPY` framework."
        )

    def add_input_constraints(self, u_bounds: Bounds):
        """
        Sets input constraints based on provided bounds.

        Args:
            u_bounds (Bounds): An instance of `Bounds` containing upper (`max_values`)
                               and lower (`min_values`) constraints for the control inputs.

        Raises:
            ValueError: If `u_bounds.max_values` and `u_bounds.min_values` have incorrect shapes.
            ValueError: If `u_bounds.max_values` is smaller than `u_bounds.min_values`.

        Special Cases:
            - To impose **no bounds** on specific inputs, set `np.inf` as the upper bound
              and `-np.inf` as the lower bound.
        """
        u_max, u_min = u_bounds.max_values, u_bounds.min_values

        if u_max.shape != (self.dims.m,) or u_min.shape != (self.dims.m,):
            raise ValueError(
                f"u_max and u_min must have shape ({self.dims.m},), " f"but got {u_max.shape} and {u_min.shape}"
            )

        if np.any(np.logical_and(np.isfinite(u_max), np.isfinite(u_min) & (u_max < u_min))):
            raise ValueError(
                f"Invalid bounds: All finite values in u_max must be >= corresponding values in u_min. "
                f"Got u_max={u_max}, u_min={u_min}."
            )

        constraints = []
        for i in range(self.ddpc_params.tau_f):
            for j in range(self.dims.m):
                idx = i * self.dims.m + j
                if np.isfinite(u_max[j]):
                    constraints.append(self.u_f_cp[idx] <= u_max[j])
                if np.isfinite(u_min[j]):
                    constraints.append(self.u_f_cp[idx] >= u_min[j])

        self.constr.extend(constraints)
        self.valid_optimization_problem = False
        self.is_unconstrained = False

    def add_terminal_output_constraints(self, y_bounds: Bounds):
        """
        Adds terminal output constraints for the final step of the prediction horizon.

        Notes:
            - Only the final predicted output in the finite future horizon is constrained.
            - Due to system lag, immediate output constraints may not always be enforceable.
            - This design choice enhances solver feasibility for a wide range of systems.
            - If stricter constraints are needed, use `add_custom_constraints`.

        Args:
            y_bounds (Bounds): A `Bounds` instance defining upper (`max_values`)
                               and lower (`min_values`) constraints for system outputs.

        Raises:
            ValueError: If `y_bounds.max_values` and `y_bounds.min_values` have incorrect shapes.
            ValueError: If any element in `y_bounds.max_values` is smaller than its corresponding `y_bounds.min_values`.

        Special Cases:
            - To impose **no bounds** on specific outputs, set `np.inf` as the upper bound
              and `-np.inf` as the lower bound.
        """
        y_max, y_min = y_bounds.max_values, y_bounds.min_values

        # Ensure y_max and y_min have correct dimensions
        if y_max.shape != (self.dims.p,) or y_min.shape != (self.dims.p,):
            raise ValueError(
                f"y_max and y_min must have shape ({self.dims.p},), " f"but got {y_max.shape} and {y_min.shape}"
            )

        # Ensure finite values of y_max are >= y_min
        if np.any(np.logical_and(np.isfinite(y_max), np.isfinite(y_min) & (y_max < y_min))):
            raise ValueError(
                f"Invalid bounds: All finite values of y_max must be >= corresponding values of y_min. "
                f"Got y_max={y_max}, y_min={y_min}"
            )

        constraints = []

        # Only constrain the final predicted output
        last_index = (self.ddpc_params.tau_f - 1) * self.dims.p  # Last time step

        for j in range(self.dims.p):
            idx = last_index + j  # Compute index

            # **Omit constraint if bound is infinite**
            if np.isfinite(y_max[j]):  # Only constrain if finite
                constraints.append(self.y_f_cp[idx] <= y_max[j])
            if np.isfinite(y_min[j]):  # Only constrain if finite
                constraints.append(self.y_f_cp[idx] >= y_min[j])

        if self.constr is None:
            self.constr = constraints
        else:
            self.constr.extend(constraints)

        self.valid_optimization_problem = False
        self.is_unconstrained = False

    def add_linear_constraints(self, A: np.ndarray, b: np.ndarray):
        """
        Adds linear inequality constraints to the optimization problem.

        The constraints are applied in the form:

            A @ [ y_f  u_f ]^T <= b

        where:
            - A is a matrix of shape (n_constraints, n_y_f + n_u_f)
            - y_f (future outputs) is of shape (n_y_f, )
            - u_f (future inputs) is of shape (n_u_f, )
            - b is a vector of shape (n_constraints, )

        Args:
            A (np.ndarray): Constraint matrix of shape (n_constraints, n_y_f + n_u_f).
            b (np.ndarray): Constraint vector of shape (n_constraints, ).

        Raises:
            ValueError: If A and b have incompatible shapes.
            TypeError: If A or b are not numpy arrays.
            Warning: If the first p columns of A are nonzero, indicating constraints on the first outputs.
        """
        if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray):
            raise TypeError("A and b must be numpy arrays.")

        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Shape mismatch: A has {A.shape[0]} rows but b has {b.shape[0]} elements.")

        expected_columns = self.y_f_cp.shape[0] + self.u_f_cp.shape[0]
        if A.shape[1] != expected_columns:
            raise ValueError(
                f"Invalid constraint dimensions: A should have {expected_columns} columns, but has {A.shape[1]}."
            )

        if np.any(A[:, : self.dims.p] != 0):
            logger.warning(
                "Constraints applied to the first predicted output `y_f`.\n"
                "   - Be aware that the **first future output (`y_f[0]`)** "
                "is usually not directly influenced by the next control input `u_f[0]`.\n"
                "   - This can lead to solver issues, often resulting in `DUAL_INFEASIBLE` errors.\n"
                "   - Consider removing constraints on `y_f[0]`."
            )

        self.constr += [A @ cp.hstack([self.y_f_cp, self.u_f_cp]) <= b]
        self.valid_optimization_problem = False
        self.is_unconstrained = False

    def reset_constraints(self) -> None:
        """
        Removes all existing constraints from the optimization problem.
        """
        self.constr.clear()
        self.valid_optimization_problem = False
        self.is_unconstrained = True

        logger.debug("Constraints has been reset.")

    def get_next_control_action(self) -> np.ndarray:
        """
        Retrieves the next computed control input.

        Returns:
            np.ndarray: The next control input of shape `(m,)`.

        Raises:
            IndexError: If the control horizon is exceeded.
            RuntimeError: If the optimization problem has not been solved or the solution is invalid.
        """
        if self.control_step >= self.ddpc_params.tau_f:
            raise IndexError(
                f"Exceeded control horizon: tau_f={self.ddpc_params.tau_f}. " f"No more control actions available."
            )

        if self._is_closed_form_possible() or self.use_mpc_cf:
            action = self.u_f[self.control_step * self.dims.m : (self.control_step + 1) * self.dims.m]
        else:
            if self.u_f_cp.value is None:
                raise RuntimeError(
                    "Optimization problem not solved or solution is invalid. Ensure `solve()` runs successfully."
                )
            action = self.u_f_cp.value[self.control_step * self.dims.m : (self.control_step + 1) * self.dims.m]

        self.control_step += 1

        return action

    def update_past_measurements(self, z_p_new: np.ndarray):
        """
        Updates the stored past measurements with new incoming data.

        Args:
            z_p_new (np.ndarray): New past measurement data (shape `(mp,)`).

        Raises:
            ValueError: If `z_p_new` does not match the expected shape.
        """
        if z_p_new.shape != (self.dims.mp,):
            raise ValueError(f"z_p_new must have shape ({self.dims.mp},), but got {z_p_new.shape}")

        # Shift past measurements to accommodate the latest data
        self.z_p_cp.value[: (self.ddpc_params.tau_p - 1) * self.dims.mp] = self.z_p_cp.value[  # type: ignore
            self.dims.mp : self.dims.n_z_p
        ]
        self.z_p_cp.value[(self.ddpc_params.tau_p - 1) * self.dims.mp : self.dims.n_z_p] = z_p_new  # type: ignore

        logger.debug("Past measurements updated.")

    def update_tracking_reference(self, y_r_new: np.ndarray, u_r_new: np.ndarray):
        """
        Updates the tracking reference trajectory.

        Args:
            y_r_new (np.ndarray): New reference trajectory for outputs (shape `(p,)`).
            u_r_new (np.ndarray): New reference trajectory for inputs (shape `(m,)`).

        Raises:
            ValueError: If `y_r_new` does not match expected dimensions `(p,)`.
            ValueError: If `u_r_new` does not match expected dimensions `(m,)`.
        """
        if y_r_new.shape != (self.dims.p,):
            raise ValueError(f"y_r_new must have shape ({self.dims.p},), but got {y_r_new.shape}")

        if u_r_new.shape != (self.dims.m,):
            raise ValueError(f"u_r_new must have shape ({self.dims.m},), but got {u_r_new.shape}")

        # Update reference trajectory over the predictive horizon
        for i in range(self.ddpc_params.tau_f):
            start_idx_y = self.dims.p * i
            end_idx_y = self.dims.p * (i + 1)
            self.y_r_cp.value[start_idx_y:end_idx_y] = y_r_new  # type: ignore

            start_idx_u = self.dims.m * i
            end_idx_u = self.dims.m * (i + 1)
            self.u_r_cp.value[start_idx_u:end_idx_u] = u_r_new  # type: ignore

        logger.debug("Tracking reference updated successfully.")

    # pylint: disable=too-many-locals
    def _construct_hankel_matrices(self, training_data: InputOutputTrajectory, n_samples: int) -> HankelMatrices:
        """
        Constructs Hankel matrices for past and future inputs/outputs.

        Args:
            training_data (InputOutputTrajectory): Trajectory data containing input-output history.
            n_samples (int): Total number of data samples.

        Returns:
            HankelMatrices

        Raises:
            ValueError: If `n_samples` is too small to form valid Hankel matrices.
        """

        tau_f, tau_p = self.ddpc_params.tau_f, self.ddpc_params.tau_p
        p, m, mp = self.dims.p, self.dims.m, self.dims.mp

        y, u = training_data.y, training_data.u
        z = np.vstack((y, u))

        n_col = n_samples - (tau_f + tau_p) + 1

        if n_col <= 0:
            raise ValueError(f"Insufficient samples: n_samples={n_samples} must be > tau_f + tau_p = {tau_f + tau_p}.")

        # Constructing past Hankel matrix
        Z_p = np.zeros((mp * tau_p, n_col))
        for i in range(tau_p):
            Z_p[i * mp : (i + 1) * mp, :] = z[:, i : i + n_col]

        # Constructing future Hankel matrices
        Y_f = np.zeros((p * tau_f, n_col))
        U_f = np.zeros((m * tau_f, n_col))
        for i in range(tau_p, tau_f + tau_p):
            Y_f[p * (i - tau_p) : p * (1 + i - tau_p), :] = y[:, i : i + n_col]
            U_f[m * (i - tau_p) : m * (1 + i - tau_p), :] = u[:, i : i + n_col]

        # Constructing complete interleaved Hankel matrix
        Z = np.zeros((mp * (tau_p + tau_f), n_col))
        for i in range(tau_p + tau_f):
            Z[i * mp : (i + 1) * mp, :] = z[:, i : i + n_col]

        # Normalize matrices
        sqrt_n_col = sqrt(n_col)
        return HankelMatrices(
            Z=Z / sqrt_n_col,
            Z_p=Z_p / sqrt_n_col,
            U_f=U_f / sqrt_n_col,
            Y_f=Y_f / sqrt_n_col,
            n_col=n_col,
            n_samples=n_samples,
        )

    def _is_closed_form_possible(self) -> bool:
        """
        Determines whether it is possible to use the closed-form solution.

        The closed-form solution can be used if:
           - The optimization problem is unconstrained (`self.is_unconstrained` is True).
           - Closed-form gain matrices (`self.cf_gains`) are available (not None).

        Returns:
            bool: True if the closed-form solution is applicable, False otherwise.
        """

        return self.is_unconstrained and self.cf_matrices is not None

    def _calculate_delta_u_f_cost(self):
        """
        Calculates the cost associated with the change in control input (Delta u_f).

        This function penalizes the difference between the last predicted control input (u_p)
        and the first future control input (u_f) using the specified weighting matrices.

        Returns:
            float or cvxpy expression: The computed cost for Delta u_f.
        """
        if self.ddpc_params.R_delta is None and self.ddpc_params.R_delta_first is None:
            return 0

        delta_uf_cost = 0

        R_d_f = getattr(self.ddpc_params, "R_delta_first", None) or self.ddpc_params.R_delta

        S_1, S_2 = self._construct_S1_S2()
        delta_uf_cost += cp.quad_form(S_2 @ self.z_p_cp - S_1 @ self.u_f_cp, R_d_f)

        if self.ddpc_params.R_delta is None:
            return delta_uf_cost

        R_d_h = self.ddpc_params.R_delta_horizon

        D = DPCUtils.construct_difference_matrix(meas_dims=self.dims.m, horizon=self.ddpc_params.tau_f)

        delta_uf_cost += cp.quad_form(self.u_f_cp, D.T @ R_d_h @ D)

        return delta_uf_cost

    def _construct_S1_S2(self):
        """
        Constructs matrices S_1 and S_2 for penalizing the difference between the last u_p and the first u_f.

        Return:
            Matrices S_1 and S_2
        """
        S_1 = np.zeros((self.dims.m, self.dims.n_u_f))
        S_2 = np.zeros((self.dims.m, self.dims.n_z_p))

        S_1[:, : self.dims.m] = np.eye(self.dims.m)  # First m columns of S1 are identity
        S_2[:, -self.dims.m :] = np.eye(self.dims.m)  # Last m columns of S2 are identity

        return S_1, S_2

    def set_state_x(self, x_new: np.ndarray):
        """
        Sets the state (`x`) for the `estimated` model.

        This method is intended to be overridden by subclasses that have a model,
        such as MPC variants.
        """
