.. _how_to_cite:

=============
Citing DNIKit
=============

The general general DNIKit publication can be cited as:

- Welsh, Megan Maher; Koski, David; Sarabia, Miguel; Sivakumar, Niv; Arawjo, Ian; Joshi, Aparna;
  Doumbouya, Moussa; Suau, Xavier; Zappella, Luca; Apostoloff, Nicholas (2023).
  `"Data and Network Introspection Kit" <https://github.com/apple/dnikit>`_;
  *https://github.com/apple/dnikit.*

.. code-block::

   @online{DNIKit,
        author = {Welsh, Megan Maher; Koski, David; Sarabia, Miguel; Sivakumar, Niv; Arawjo, Ian; Joshi, Aparna; Doumbouya, Moussa; Suau, Xavier; Zappella, Luca; Apostoloff, Nicholas},
        title = {Data and Network Introspection Kit},
        year = 2023,
        url = {https://github.com/apple/dnikit},
   }

In addition, there are possible additional citations to include for each specific introspector
(algorithm) that was used. These citations are listed below.

Visualizing the Dataset Report, Familiarity, Duplicates, or Projection (Dimension Reduction) with `Symphony UI <https://github.com/apple/ml-symphony>`_:
 - Bäuerle, Alex, Ángel Alexander Cabrera, Fred Hohman, Megan Maher, David Koski, Xavier Suau, Titus Barik, and Dominik Moritz.
   `"Symphony: Composing Interactive Interfaces for Machine Learning." <https://dl.acm.org/doi/abs/10.1145/3491102.3502102>`_
   In *CHI Conference on Human Factors in Computing Systems,* pp. 1-14. 2022.
PFA:
 - Cuadros, Xavier Suau; Zappella, Luca; Apostoloff, Nicholas.
   `"Filter distillation for network compression." <https://arxiv.org/abs/1807.10585>`_
   In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision,* pp. 3140-3149. 2020.
DimensionReduction Strategy ``TSNE``:
  - This is not a DNIKit citation, but here is the reference for TSNE:
      - Van der Maaten, L.J.P.; Hinton, G.E. (Nov 2008). "Visualizing Data Using t-SNE" (PDF). Journal of Machine Learning Research. 9: 2579–2605.
DimensionReduction Strategy ``UMAP``:
  - This is not a DNIKit citation, but here is the reference for UMAP:
      - McInnes, Leland; Healy, John; Melville, James (2018-12-07). "Uniform manifold approximation and projection for dimension reduction". arXiv:1802.03426.
DimensionReduction Strategy ``PacMAP``:
  - This is not a DNIKit citation, but here is the reference for PacMAP:
      - Yingfan Wang, Haiyang Huang, Cynthia Rudin, & Yaron Shaposhnik (2021).
        Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization.
        Journal of Machine Learning Research, 22(201), 1-73.
Duplicates:
  - This is not a DNIKit citation, but here is the reference for ANNOY:
      - Bernhardsson, Erik (2018); "Annoy: Approximate Nearest Neighbors Oh Yeah in C++/Python";
        *https://pypi.org/project/annoy/*.
IUA:
 - No additional expected citation for this introspector
Familiarity (no vis):
 - No additional expected citation for this introspector without use of Symphony visualization (see earlier citations)

Example of citing DNIKit
------------------------

For instance, when using both :ref:`Familiarity analysis <familiarity>`
and :ref:`PFA for compression <network_compression>`, the following citations are appropriate:

1. the main reference to DNIKit, at the top of this page (Welsh et al. 2023)
2. Bäuerle et al. 2022 for Familiarity,
3. Cuadros et al. 2020 for PFA
