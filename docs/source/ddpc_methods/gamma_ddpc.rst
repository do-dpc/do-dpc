γ-DPC
======

γ-DPC is a **DPC** method presented in the following papers:

- `Uncertainty-aware Data-Driven Predictive Control in a Stochastic Setting <https://arxiv.org/pdf/2211.10321>`_
- `Data-Driven Predictive Control in a Stochastic Setting: A Unified Framework <https://www.sciencedirect.com/science/article/pii/S0005109823001139>`_


Parameters and Variables
------------------------

L is the LQ decomposition of the Hankel Matrix :math:`[Z_p \ U_f \ Y_f]^T`.

.. math::

    L =
    \begin{bmatrix}
    L_{11} & 0      & 0      \\
    L_{21} & L_{22} & 0      \\
    L_{31} & L_{32} & L_{33}
    \end{bmatrix}

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Symbol
     - Dimensions
   * - :math:`L`
     - :math:`(n\_z_p + n\_u_f + n\_y_f) \times (n\_z_p + n\_u_f + n\_y_f)`
   * - :math:`L_{11}`
     - :math:`n\_z_p \times n\_z_p`
   * - :math:`L_{21}`
     - :math:`n\_u_f \times n\_z_p`
   * - :math:`L_{22}`
     - :math:`n\_u_f \times n\_u_f`
   * - :math:`L_{31}`
     - :math:`n\_y_f \times n\_z_p`
   * - :math:`L_{32}`
     - :math:`n\_y_f \times n\_u_f`
   * - :math:`L_{33}`
     - :math:`n\_y_f \times n\_y_f`


Optimization Formulation
------------------------

.. math::

    \min_{u_f,y_f} &\quad \|y_f - y_r\|_Q^2 + \|u_f - u_r\|_R^2 + \lambda_{\gamma,2}||\gamma_2||_2^2 \\
    \text{s.t.} &\quad u_f = L_{21} \gamma_1 + L_{22} \gamma_2 \\
     &\quad y_f = L_{31} \gamma_1 + L_{32} \gamma_2 \\
    &\quad \gamma_1^\star = L_{11}^{-1} z_p \\
     &\quad u_f \in \mathcal{U}, \quad y_f \in \mathcal{Y}

Closed-Form Solution Derivation
-------------------------------

Solving for :math:`\gamma_2`

.. math::

   \gamma_2 = -L_{22}^{-1}L_{21}L_{1}^{-1}z_p+L_{22}^{-1}u_f

Defining helper matrices for readability:

.. math::

   \begin{align*}
   T_1 &:= L_{22}^{-1}L_{21}L_{1}^{-1} \\
   T_2 &:= L_{22}^{-1}
   \end{align*}

Therefore:

.. math::

   \gamma_2 = -T_1z_p+T_2u_f

Replacing :math:`\gamma_1`, :math:`\gamma_2`:

.. math::

   y_f = (L_{31}L_{1}^{-1} - L_{32}T_1)z_p + L_{32}T_2 u_f

Defining helper matrices for readability:

.. math::

   \begin{align*}
   T_3 &:= (L_{31}L_{1}^{-1} - L_{32}T_1) \\
   T_4 &:= L_{32}T_2
   \end{align*}

Therefore:

.. math::

   y_f = T_3z_p +T_4 u_f

The cost function can be written as:

.. math::

   \|T_3z_p+T_4u_f-y_r\|_Q^2 + \|u_f-u_r\|_R^2 + \lambda_{\gamma,2}\|-T_1z_p+T_2u_f\|^2_2

Taking the derivative with respect to :math:`u_f` and setting it to zero leads to:

.. math::

   (T_4^TQT_4+R+\lambda_{\gamma,2}T_2^TT_2)u_f^\star = (T_2^TT_1-T_4^TT_3)z_p+T_4^TQy_r+Ru_r

Defining helper matrices:

.. math::

   \begin{align*}
   F_1 &=  (T_4^TQT_4+R+\lambda_{\gamma,2}T_2^TT_2)^{-1} \\
   F_2 &= (\lambda_{\gamma,2}T_2^TT_1-T_4^TQT_3) \\
   F_3 &= T_4^TQ
   \end{align*}

Leads to the following closed-form gain matrices:

.. math::

   K_{z_p} &= F_1 F_2 \\
   K_{y_r} &= F_1 F_3 \\
   K_{u_r} &= F_1 R

With the optimal :math:`u_f^\star` calculated as:

.. math::

   u_f^* = K_{z_p} z_p + K_{y_r} y_r + K_{u_r} u_r
