# Time Evolution by Tensor Network
Hands-on of the time evolution of a quantum many-body system by tensor network methods

## Slide for basics of tensor network simulation
Slide of main lecture: "[Quantum simulation by tensor network methods](slide/SQAI20250807.pdf)"

## Notebooks
Here we share notebooks for the hands-on tutorial on time evolution methods by tensor networks.

- Time evolution of one-dimensional systems by TEBD of MPS: [Ex_TE.ipynb](notebook/Ex_TE.ipynb)[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TsuyoshiOkubo/time-evolution-by-tensor-network/blob/main/notebook/Ex_TE.ipynb)
- Time evolution of two-dimensional systems by TEBD of MPS and PEPS: [Ex_TE_2d.ipynb](notebook/Ex_TE_2d.ipynb)[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TsuyoshiOkubo/time-evolution-by-tensor-network/blob/main/notebook/Ex_TE_2d.ipynb)
- Time evolution of infinite systems by iMPS and iPEPS: [Ex_TE_infinite.ipynb](notebook/Ex_TE_infinite.ipynb)[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TsuyoshiOkubo/time-evolution-by-tensor-network/blob/main/notebook/Ex_TE_infinite.ipynb)


## Information of TeNeS
TeNeS is an open-source software for simulations of two-dimensional lattice models by iTPS (iPEPS).
- [GitHub](https://github.com/issp-center-dev/TeNeS)
- [Official web page](https://www.pasums.issp.u-tokyo.ac.jp/tenes/en)
- Papers
    - Y. Motoyama, T. Okubo, K. Yoshimi, S. Morita, T. Aoyama, T. Kato, and N. Kawashima, [Comput. Phys. Commun. 315 109692(2025)](https://doi.org/10.1016/j.cpc.2025.109692)
    - Y. Motoyama, T. Okubo, K. Yoshimi, S. Morita, T. Kato, and N. Kawashima, [Comput. Phys. Commun. 279, 108437 (2022).](https://doi.org/10.1016/j.cpc.2022.108437)
    
We can simulate 
- Ground states
- Real and Imaginary time evolutions
- Finite temperature properties

Instead of installing TeNeS, we can use it, for example, from [MateriApps Live!](https://github.com/cmsi/MateriAppsLive). If you try it, please prepare your (virtual) Linux environment by MateriApps Live as

* [Virtual box](https://github.com/cmsi/MateriAppsLive/wiki/GettingStartedOVA-en)
* [Docker](https://github.com/cmsi/MateriAppsLive/wiki/GettingStartedDocker-en)
