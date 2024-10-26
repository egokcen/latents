.. _gfa_user_guide:

Group factor analysis (GFA)
===========================

Group factor analysis (GFA) is a Bayesian dimensionality reduction approach 
intended for multi-view or multi-group data. From the data, it automatically 
determines (1) how many latent variables are needed to describe the interactions
across all observed groups, and (2) for each latent variable, which subset of 
groups is involved.

`latents` implements a version of GFA with anisotropic observation noise,
used in the following `paper <https://neurips.cc/virtual/2023/poster/70171>`_:

  - Gokcen, E., Jasper, A. I., Xu, A., Kohn, A., Machens, C. K. & Yu, B. M. 
    Uncovering motifs of concurrent signaling across multiple neuronal 
    populations. *Advances in Neural Information Processing Systems* **36**,
    34711-34722 (2023).

An original formulation with isotropic observation noise can be found
`here <https://doi.org/10.1109/tnnls.2014.2376974>`_:

  - Klami, A., Virtanen, S., Leppäaho, E. & Kaski, S. Group Factor Analysis. 
    *IEEE Transactions on Neural Networks and Learning Systems* **26**,
    2136--2147 (2015).

For a comprehensive demonstration of how to use the `gfa` subpackage, see the
:doc:`../demo_gallery/demo_gfa` notebook
