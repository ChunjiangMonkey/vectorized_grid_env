# Vectorized Mix-motivate MARL Environments Based on Pytorch
Three vectorized mix-motivate MARL environments, including **Coin Game**, **Gathering**, and **Harvest**.
Their implementations comply with the Gym API.

## Main components
* `coin_game.py`: Vectorized implementation of Coin Game, which is proposed by [Lerer and Peysakhovich](https://arxiv.org/pdf/1707.01068.pdf).
* `gathering.py`: Vectorized implementation of Gathering, which is proposed by [Leibo et al.](https://arxiv.org/pdf/1702.03037.pdf).
* `harvest.py`: Vectorized implementation of Simple Harvest, which is proposed by [Willis and Luck](https://kclpure.kcl.ac.uk/ws/portalfiles/portal/207013371/ALA2023_paper_65.pdf).
* `main.ipynb`: Providing some simple examples on how to use these environments.

## Setup
Just install the latest version of pytorch (tested on pytorch 2.1.2).

## Acknowledgements
* The implementation of Coin Game comes from https://github.com/luchris429/model-free-opponent-shaping.
Referring to Coin Game, we implement Gathering and Harvest.
* [Hao Li](https://github.com/ynulihao) has fixed a large number of bugs.





