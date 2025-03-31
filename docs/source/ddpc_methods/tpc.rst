TPC
===

Transient Predictive Control (TPC) is a **DPC** method with the following key properties:

- Small online optimization problem,
- Predictors are causal, and
- not biased when closed-loop training data is used.

TPC is presented in the following research papers:

- `The Transient Predictor <https://www.research-collection.ethz.ch/handle/20.500.11850/716622>`_
- `On the Impact of Regularization in Data-Driven Predictive Control <https://arxiv.org/abs/2304.00263>`_



Parameters and Variables
------------------------

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Symbol
     - Definition
     - Dimensions
   * - :math:`H_p,\ H_u`
     - Multistep predictor
     - :math:`n\_y_f \times n\_z_p`; :math:`n\_y_f \times n\_u_f`
   * - :math:`\Lambda_{uu},\ \Lambda_{uz},\ \Lambda_{uy}`
     - Regularization matrices
     - :math:`n\_u_f \times n\_u_f`; :math:`n\_u_f \times n\_z_p`; :math:`n\_u_f \times n\_y_f`



Optimization Formulation
------------------------

.. math::

    \min_{u_f,y_f} &\quad \|y_f - y_r\|_Q^2 + \|u_f - u_r\|_R^2 + r(u_f, y_r, z_p) \\
    \text{s.t.} &\quad y_f = H_u u_f + H_p z_p \\
     &\quad u_f \in \mathcal{U}, \quad y_f \in \mathcal{Y}

where:

.. math::

    r(u_f, y_r, z_p) :=  u_f^T \Lambda_{uu}u_f+2u^T_f \Lambda_{uz}z_p+2u_f^T \Lambda_{uy}y_r



Closed-Form Solution Derivation
-------------------------------

Set :math:`y_f` into the cost equation leads to:

.. math::

   \|H_u u_f + H_p z_p - y_r\|_Q^2 + \|u_f - u_r\|_R^2 + u_f^T \Lambda_{uu}u_f+2u^T_f \Lambda_{uz}z_p+2u_f^T \Lambda_{uy}y_r

Now taking the derivative in respect to :math:`u_f` and set it to zero.

.. math::

   (H_u ^T Q H_u + R + \Lambda_{uu})u_f^\star = (-H_u ^TQH_p -\Lambda_{uz})z_p + (H_u^TQ - \Lambda_{uy})y_r + R u_r

Defining the help matrices:

.. math::

   F_1 &= (H_u ^T Q H_u + R + \Lambda_{uu})^{-1} \\
   F_2 &= -H_u ^TQH_p -\Lambda_{uz} \\
   F_3 &= H_u^TQ - \Lambda_{uy}

Leads to the following closed-form gain matrices:

.. math::

   K_{z_p} &= F_1 F_2 \\
   K_{y_r} &= F_1 F_3 \\
   K_{u_r} &= F_1 R

With the optimal :math:`u_f^\star` calculated as:

.. math::

   u_f^* = K_{z_p} z_p + K_{y_r} y_r + K_{u_r} u_r
