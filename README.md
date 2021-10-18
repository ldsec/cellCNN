# Private-CellCnn

Private-CellCnn is a Golang library that ensures CellCnn[1] analysis in privacy-preserving, distributed, N-party setting.

[1] E. Arvaniti and M. Claassen. Sensitive detection of rare disease-associated cell subsets via representation learning.Nat Commun, 8:1–10, 2017

## Overview
- `cellCNN/cellCNN_clear`: contains the relevant data preprocessing and distribution scripts and centralized and distributed cellCnn implementation for testing/benchmarking.
    - Data generation and preprocessing scripts require Python 3.6
- `cellCNN/cellCNN_optimized`: contains the optimized, distributed, privacy-preserving implementation of cellCnn with multiparty homomorphic encryption
- `cellCNN/cellcnnPoseidon`: contains the distributed, privacy-preserving implementation of cellCnn with multiparty homomorphic encryption, relying on POSEIDON[2] packing strategies, for benchmarking

[2] S. Sav, A. Pyrgelis, J. R. Troncoso-Pastoriza, D. Froelicher, J.-P. Bossuat,
  J. S. Sousa, and J.-P. Hubaux, “POSEIDON: Privacy-Preserving Federated
  Neural Network Learning,” in NDSS, 2021






  
