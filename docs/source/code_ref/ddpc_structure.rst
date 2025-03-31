DPC Structures
===============

Calculated by DPC specific algorithm:

- DPCPredictorMatrices
- DPCRegularization Matrices
- DPCClosedFormSolutionMatrices

User Input:

- DPCParameters

Calculated:

- DPCDimensions

DPC Class
==========

Arguments
---------

- ddpc_parameters (DPCParameters)
- training_data (InputOutputTrajectory)

Attributes
----------

- ddpc_params
- ddpc_dims

Matrices:

- hankel_matrices
- pred_matrices
- reg_matrices
- cf_matrixes

CVXPY:

- constr
- cost
- problem
- valid_opt_problem

CP Variables:

- u_f_cp
- y_f_cp

CP Parameters

- y_r_cp
- u_r_cp
- z_p_cp

DPC Init
---------

1. Check valid training data, parameters
2. Calculate dimensions
3. Construct Hankel matrices
4. calculate pred, reg, cf matrices
5. Initialize CVXPY optimization problem

Class Methods
-------------

- register
- instantiate

Abstract methods
----------------

Calculate:

- calculate_predictor_matrices
- calculate_regularization_matrices
- calculate_closed_form_solution_matrices

Get expression:

- get_regularization_cost_expression
- get_predictor_constraint_expression

Concrete methods
----------------

Optimization:

- build_opt_problem
- solve
- constraint handling
- get next control action

Update:

- update_past_measurements
- update_reference_trajectory
