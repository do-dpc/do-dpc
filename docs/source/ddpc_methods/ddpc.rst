DPC
====

Data-Driven Predictive Control (DPC) is similar to Model Predictive Control (MPC) but relies on collected data
instead of an explicit model for future measurement predictions.
Different methods exist for implementing this approach, with the following currently available:

- :doc:`TPC <tpc>`
- :doc:`DeePC <deepc>`
- :doc:`γ-DPC <gamma_ddpc>`
- :doc:`SPC <spc>`
- :doc:`Oracle MPC <mpc>`
- :doc:`n4sid MPC  <mpc>`

The **Oracle MPC** utilizes the true system model, serving as a ground truth benchmark—-no method can outperform it.


Parameters and Variables
------------------------

The following subscripts has been used:

- :math:`_p` referencing the past values,
- :math:`_f` referencing the future values,
- :math:`_r` referencing the future reference values

The following table summarizes the key symbols used in the general **DPC**:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Symbol
     - Definition
   * - :math:`y`
     - System outputs
   * - :math:`u`
     - Control inputs
   * - :math:`y_r`, :math:`u_r`
     - Reference trajectory
   * - :math:`y_f`, :math:`u_f`
     - Future inputs/outputs
   * - :math:`z_p`
     - Past inputs/outputs
   * - :math:`m`
     - Number of control inputs
   * - :math:`p`
     - Number of system outputs
   * - :math:`Q`, :math:`R`
     - Cost weight matrices
   * - :math:`\tau_p`, :math:`\tau_f`
     - Past/future time horizon

State Vector Representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^


The state vector :math:`z(t)` is defined as:

.. math::

   z(t) = \begin{bmatrix} y(t) \\ u(t) \end{bmatrix}


Dimensions
^^^^^^^^^^

The dimensions of the parameters are calculated as follows:

.. math::
   n\_y_f &= p  \tau_f, \\
   n\_u_f &= m \tau_f,  \\
   n\_z_p &= (m+p) \tau_p


Hankel Matrices
---------------

The trajectory of :math:`y, u` are interleaved written as :math:`y`.

The following Hankel matrices have been defined:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Symbol
     - Definition
   * - :math:`Z`
     - Future and past input/output trajectories interleaved.
   * - :math:`Z_p`
     - Past input/output trajectories interleaved.
   * - :math:`Y_f`
     - Future output trajectories.
   * - :math:`U_f`
     -  Future input trajectories.
   * - :math:`n_{\text{samples}}`
     -  Number of samples.
   * - :math:`n_{\text{col}}`
     -  Number of entries in a column.

:math:`n_{\text{col}}` is calculated the following way:

.. math::
  n_{\text{col}} = n_{\text{samples}} - \tau_f - \tau_p +1


Implementation
--------------

DPC implementation is divided into three parts:

1. **Data Collection** –  The input/output training data is gathered offline, data needs to be persistently exited
2. **Offline Calculations** –
    Precomputing necessary matrices and optimization parameters, which do not depend on :math:`z_p,\ y_r,\ u_r`
3. **Online Optimization** – Solving a constrained optimization problem in real-time to determine the control action.

The **Online Optimization**  has the following structure:

.. math::

    \min_{u_f, \cdot} &\quad \|y_f - y_r\|_Q^2 + \|u_f - u_r\|_R^2 + r(\cdot) \\
    \text{s.t.} &\quad y_f = f(\cdot) \\
    &\quad u_f \in \mathcal{U}, \quad y_f \in \mathcal{Y}

where :math:`r(\cdot)`, :math:`f(\cdot)` are linear functions
which utilize matrices calculated in the **Offline Calculations**.
Additional slack decision variables can be introduced.
:math:`r(\cdot)`, :math:`f(\cdot)` differ between `DPC` methods.

**Disclaimer:** There is sometimes a possibility to incorporate prior knowledge
or some model information,
but this has not been implemented in this library.

Tunable Parameters
^^^^^^^^^^^^^^^^^^


For the cost, there is a possibility to introduce tunable parameters, denoted by :math:`\lambda_{\_}`.
Those tunable parameters either need hand tuning
or can be (sometimes) tuned automatically with some sort of model information.


Closed-Form Solution
--------------------

If the problem is unconstrained, i.e., :math:`\mathcal{U} \in \mathbb{R}^{m \tau_f}` and
:math:`\mathcal{Y} \in \mathbb{R}^{p \tau_f}`, and the problem consists only of a quadratic and linear cost,
then a closed-form solution exists.

The closed-form solution is derived by:

1. Substituting the equality constraints into the cost function.
2. Differentiating with respect to :math:`u_f`.
3. Setting the derivative to zero.


This closed-form solution has the form:

.. math::

   u_f^* = K_{z_p} z_p + K_{y_r} y_r + K_{u_r} u_r

where :math:`K_{z_p}, K_{y_r}, K_{u_r}` are gain matrices computed using the specific DPC-method and the collected data.

Delta u_f cost
--------------

A :math:`\Delta u_f` cost with the weight matrix :math:`R_\Delta` can be added for the difference in controller input.
Where :math:`R_\Delta^{(0)}` applies to the difference between the last measured :math:`u_p` and the first :math:`u_f`,
which significantly impacts the transition and can be set independently.

.. note::
    Adding a :math:`\Delta u_f` cost results in a smoother transition between reference points, reducing overshoot.
    However, it typically leads to a slower response.


If :math:`R_\Delta^{(0)}` is not specified, the library defaults to using :math:`R_\Delta`.


Implementation
^^^^^^^^^^^^^^

.. math::

   u_f = \begin{bmatrix}
   u_1^{(1)} \\
   u_2^{(1)} \\
   \vdots \\
   u_1^{(\tau)} \\
   u_2^{(\tau)}
   \end{bmatrix}, \quad
   \Delta u_f = \begin{bmatrix}
   u_1^{(1)} - u_1^{(2)}  \\
   u_2^{(1)} - u_2^{(2)} \\
   \vdots \\
   u_1^{(\tau-1)} - u_1^{(\tau)}\\
   u_2^{(\tau-1)} - u_2^{(\tau)}
   \end{bmatrix}

One can rewrite it with the difference matrix `D`.


.. math::

   \Delta u_f = D u_f, \ D \in \mathbb{R}^{m(\tau-1)\times m \tau}

The cost :math:`R_\Delta` is a diagonal matrix.

Leading to the cost:

.. math::

   u_f^T D^T R_\Delta D u_f.

An additional cost for the difference of the last :math:`u_p` and the first :math:`u_f`.


.. math::

   (u_p^{(0)}-u_f^{(0)})^T R_\Delta^{(0)} (u_p^{(0)}-u_f^{(0)})

With defining :math:`S_1, \ S_2` s.t.:

.. math::

   u_f^{(0)} &= S_1 u_f \\
   u_p^{(0)} &= S_2 z_p \\
   \text{where} & \quad S_1 \in \mathbb{R}^{m \times m \tau_f}, S_2 \in \mathbb{R}^{m \times (m+p) \tau_p}

We get the following cost:

.. math::

   (S_2 z_p - S_1 u_f)^T  R_\Delta^{(0)} (S_2 z_p - S_1 u_f)
