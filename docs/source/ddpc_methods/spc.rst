SPC
===

Subspace Predictive Control (SPC)  is a **DPC** method.

SPC is presented in the following research paper:

- `SPC: Subspace Predictive Control <https://www.sciencedirect.com/science/article/pii/S1474667017566835>`_

Parameters and Variables
------------------------

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Symbol
     - Definition
     - Dimensions
   * - :math:`S_p,\ S_u`
     - Subspace predictor
     - :math:`n\_y_f \times n\_z_p`; :math:`n\_y_f \times n\_u_f`


Optimization Formulation
------------------------

.. math::

    \min_{u_f,y_f} &\quad \|y_f - y_r\|_Q^2 + \|u_f - u_r\|_R^2 \\
    \text{s.t.} &\quad y_f = S_u u_f + S_p z_p \\
     &\quad u_f \in \mathcal{U}, \quad y_f \in \mathcal{Y}

The Subspace Multistep Predictor :math:`S` is calculated as follow:

.. math::

   S &:= Y_f \begin{bmatrix} Z_p \\ U_f \end{bmatrix}^\dagger \\
     &= Y_f \begin{bmatrix} Z_p \\ U_f \end{bmatrix}^T
     \left(\begin{bmatrix} Z_p \\ U_f \end{bmatrix} \begin{bmatrix} Z_p \\ U_f \end{bmatrix}^T \right)^{-1}


Closed-Form Solution Derivation
-------------------------------

Set :math:`y_f` into the cost equation leads to:

.. math::

   \|S_u u_f + S_p z_p - y_r\|_Q^2 + \|u_f - u_r\|_R^2

Now taking the derivative in respect to :math:`u_f` and set it to zero.

.. math::

   (S_u ^T Q S_u + R )u_f^\star = (-S_u ^TQS_p)z_p + (S_u^TQ)y_r + R u_r

Defining the help matrices:

.. math::

   F_1 &= (S_u ^T Q S_u + R)^{-1} \\
   F_2 &= -S_u ^TQS_p \\
   F_3 &= S_u^TQ

Leads to the following closed-form gain matrices:

.. math::

   K_{z_p} &= F_1 F_2 \\
   K_{y_r} &= F_1 F_3 \\
   K_{u_r} &= F_1 R

With the optimal :math:`u_f^\star` calculated as:

.. math::

   u_f^* = K_{z_p} z_p + K_{y_r} y_r + K_{u_r} u_r
