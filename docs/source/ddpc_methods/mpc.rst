MPC
===

This method requires a model or model estimation of the system along with a Kalman Gain matrix.
The process noise and measurement noise covariance matrices need to be provided or estimated.

For more details on the MPC method, refer to the
`Wikipedia page <https://en.wikipedia.org/wiki/Model_predictive_control>`_.

We have two variants implemented to obtain the model:

1. Oracle
2. N4SID

Oracle
------

The Oracle MPC is intended to be used as a benchmark.
In this variant, the true state model (or linearized dynamics) is injected.
The optimal Kalman gain is calculated. No other DPC method can outperform this variant.

N4SID
-----

The Numerical Subspace State Space System Identification (N4SID or NFourSID) is a system identification algorithm,
detailed in this `paper <https://www.sciencedirect.com/science/article/pii/0005109894902305>`_.
The tunable parameter ``n_block_rows`` :math:`\geq 1` must be an integer.

The state dimension :math:`n_{state}` is calculated as:

.. math::

    n_{state} = p * n_\text{block_rows},

where :math:`p` is the number of outputs.

.. note::
    Choosing a large ``n_block_rows`` increases computational complexity,
    while a small ``n_block_rows`` may hinder the accurate determination of the system order in the eigenvalue diagram.

Optimization Formulation
------------------------

.. math::

    \min_{u_f,y_f} &\quad \|y_f - y_r\|_Q^2 + \|u_f - u_r\|_R^2 \\
    \text{s.t.} &\quad y_f = \Gamma x +  H_u u_f \\
     &\quad u_f \in \mathcal{U}, \quad y_f \in \mathcal{Y}

Here:

- :math:`\Gamma` is the extended observability matrix,
- :math:`H_u` maps future inputs to future outputs,
- :math:`x` is the estimated state of the system which needs to be updated.

:math:`\Gamma` is constructed as follow:

.. math::

    \Gamma = \begin{bmatrix} C \\ CA \\ CA^2 \\ \vdots \\ CA^{(\tau_f-1)} \end{bmatrix}

:math:`H_u` is constructed as follow:

.. math::

    H_u = \begin{bmatrix}
    D & 0 & 0 & \cdots & 0 \\
    CB & D & 0 & \cdots & 0 \\
    CAB & CB & D & \cdots & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    CA^{(\tau_f-2)}B & CA^{(\tau_f-3)}B &  CA^{(\tau_f-4)}B& \cdots  & D
    \end{bmatrix}


Solving
-------

The solution is set up in two steps:

1. Update the state using the Kalman Filter.

.. math::

    x &= A x + B u \\
    x &= x + K(y-Cx-Du)


2. Solve the optimization problem


Closed-Form Solution Derivation
-------------------------------

Set :math:`y_f` into the cost equation leads to:

.. math::

   \|H_u u_f + \Gamma x - y_r\|_Q^2 + \|u_f - u_r\|_R^2

Now taking the derivative in respect to :math:`u_f` and set it to zero.

.. math::

   (H_u^T Q H_u + R )u_f^\star = (-H_u^TQ \Gamma)x + (H_u^TQ)y_r + R u_r

Defining the help matrices:

.. math::

   F_1 &= (H_u^T Q H_u + R)^{-1} \\
   F_2 &= -H_u^TQ \Gamma \\
   F_3 &= H_u^TQ

Leads to the following closed-form gain matrices:

.. math::

   K_{x} &= F_1 F_2 \\
   K_{y_r} &= F_1 F_3 \\
   K_{u_r} &= F_1 R

With the optimal :math:`u_f^\star` calculated as:

.. math::

   u_f^* = K_{x} x + K_{y_r} y_r + K_{u_r} u_r
