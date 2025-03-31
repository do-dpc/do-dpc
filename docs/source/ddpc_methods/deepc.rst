DeePC
=====

Data-Enabled Predictive Control (DeePC) is a **DPC** method presented in the following papers:

- `Data-Enabled Predictive Control: In the Shallows of the DeePC <https://ieeexplore.ieee.org/document/8795639>`_
- `Regularized and Distributionally Robust Data-Enabled Predictive Control <https://ieeexplore.ieee.org/document/9028943>`_


Performance Considerations
--------------------------

This method differs from other DPC approaches by optimizing over a slack decision variable :math:`g` of size
:math:`n_{\text{col}}`. Since :math:`n_{\text{col}} \approx n_{\text{samples}}` for large
:math:`n_{\text{samples}}`, computational performance is directly impacted by :math:`n_{\text{samples}}`,
leading to a computational complexity of :math:`\mathcal{O}(n_{\text{samples}}^2)`.

With a large number of samples, performance further degrades due to memory limitations.


Performance Benchmark (MacBook Pro, Double Integrator):

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Number of Samples
     - Total Simulation Time
   * - 100
     - ~1 second
   * - 300
     - ~10 seconds
   * - 1000
     - ~4 minutes

.. note::
   100 samples are generally insufficient for the double integrator when process and measurement noise are present.



Noise Rejection
---------------

In the presence of noise, an additional slack variable :math:`\sigma` is required to ensure feasibility of the
equality constraints.

.. math::

   [Z_p] g = [z_p] + \sigma

where:

.. math::

   \sigma = \begin{bmatrix} \sigma_1 \\ 0 \\ \sigma_2 \\ \dots \\ \sigma_{\tau,f} \\ 0 \end{bmatrix}

which is penalized by the cost function:

.. math::

   \lambda_{\sigma} \|\sigma\|_2^2

where :math:`\lambda_{\sigma}` should be chosen large.

This allows the past measurements to deviate slightly from the Hankel matrices.

Additionally, a cost term is needed to regulate the size of :math:`g` for a noise robust DeePC.

For further details, refer to
`Regularized and Distributionally Robust Data-Enabled Predictive Control <https://ieeexplore.ieee.org/document/9028943>`_.


Optimization Formulation
------------------------

.. math::

    \min_{g, \sigma} &\quad [\|y_f - y_r\|_Q^2 + \|u_f - u_r\|_R^2 \\
   &\quad + \lambda_{g,1} ||g||_1 + \lambda_{g,2} ||g||_2^2 \\
   &\quad + \lambda_p ||(I-\Pi)g||_2^2 + \lambda_\sigma ||\sigma||_2^2 ]\\
    \text{s.t.} &\quad \begin{bmatrix} Z_p \\ U_f \\ Y_f \end{bmatrix} g =
    \begin{bmatrix} z_p + \sigma \\ u_f \\ y_f \end{bmatrix}\\
     &\quad u_f \in \mathcal{U}, \quad y_f \in \mathcal{Y}

where :math:`\Pi` is the Kernel Projection matrix calculated as follow:

.. math::

   \Pi &:= \begin{bmatrix} Z_p \\ U_f \end{bmatrix}^\dagger \begin{bmatrix} Z_p \\ U_f \end{bmatrix} \\
     &= \begin{bmatrix} Z_p \\ U_f \end{bmatrix}^T
     \left( \begin{bmatrix} Z_p \\ U_f \end{bmatrix} \begin{bmatrix} Z_p \\ U_f \end{bmatrix}^T \right)^{-1}
    \begin{bmatrix} Z_p \\ U_f \end{bmatrix}


Closed-Form Solution Derivation
-------------------------------

Since the derivative of the 1-norm is not well-defined everywhere, :math:`\lambda_{g,1}` must be set to zero for
the derivation to be valid.

From the equality constraints, we obtain:

.. math::

    \sigma &= Z_p g - z_p \\
    y_f &= Y_f g \\
    u_f &= U_f g

Substituting these into the cost function gives:

.. math::

   ||Y_fg-y_r||_Q^2+ ||U_fg-u_r||^2_R
   + \lambda_{g,2}||g||_2^2
   + \lambda_p||(I-\Pi)g||_2^2
   + \lambda_\sigma||Z_pg-z_p||_2^2

Taking the derivative with respect to :math:`g` and setting it to zero:

.. math::

   (Y_f^TQY_f+U_f^TRU_f+\lambda_{g,2}+\lambda_p(I-\Pi)^T(I-\Pi)+\lambda_\sigma Z_p^T Z_p) g^\star
   = Y_f^TQy_r+U_f^TRu_r+\lambda_\sigma Z_p^T z_p

For readability:

.. math::
    T_1 &:= Y_f^TQY_f+U_f^TRU_f \\
    T_2 &:=\lambda_{g,2}+\lambda_p(I-\Pi)^T(I-\Pi) \\
    T_3 &:= \lambda_\sigma Z_p^T Z_p


Replacing :math:`g^\star = U_f^{-1} u_f^\star`, we obtain the following helper matrices:


.. math::

   F_1 &:= U_f (T_1+T_2+T_3)^{-1} \\
   F_2 &:= \lambda_\sigma Z_p^T \\
   F_3 &:= Y_f^TQ \\
   F_4 &:= U_f^TR

This leads to the following closed-form gain matrices:

.. math::

   K_{z_p} &= F_1 F_2 \\
   K_{y_r} &= F_1 F_3 \\
   K_{u_r} &= F_1 F_4

With the optimal :math:`u_f^\star` calculated as:

.. math::

   u_f^* = K_{z_p} z_p + K_{y_r} y_r + K_{u_r} u_r
