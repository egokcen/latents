.. _mdlag_user_guide:

Delayed latents across multiple groups (mDLAG)
================================================

Delayed latents across multiple groups (mDLAG) is a Bayesian dimensionality reduction
approach intended for multi-view or multi-group time series data. From the data, it
automatically determines (1) the subset of groups described by each latent dimension,
(2) the direction of signal flow among those groups, and (3) how those signals evolve
over time within and across trials.

For mathematical details, see the following
`paper <https://neurips.cc/virtual/2023/poster/70171>`_:

  - Gokcen, E., Jasper, A. I., Xu, A., Kohn, A., Machens, C. K. & Yu, B. M.
    Uncovering motifs of concurrent signaling across multiple neuronal
    populations. *Advances in Neural Information Processing Systems* **36**,
    34711-34722 (2023).

For a comprehensive demonstration of how to use the `mdlag` subpackage, see the
:doc:`../demo_gallery/demo_mdlag` notebook
