This repository includes the implementation of the paper PriCell (accepted at Patterns). 
Please use the following BibTex entry for citing PriCell:

	@Article{Sav2022,
		author={Sav, Sinem and Bossuat, Jean-Philippe and Troncoso-Pastoriza, Juan R. and Claassen, Manfred and Hubaux, Jean-Pierre},
		title={Privacy-preserving federated neural network learning for disease-associated cell classification},
		journal={Patterns},
		year={2022},
		month={May},
		day={13},
		publisher={Elsevier},
		volume={3},
		number={5},
		issn={2666-3899},
		doi={10.1016/j.patter.2022.100487},
		url={https://doi.org/10.1016/j.patter.2022.100487}
	}


# PriCell (Privacy-Preserving CellCNN)

PriCell (private version) is a Golang library that ensures CellCnn[1] analysis in privacy-preserving, distributed, N-party setting.

[1] E. Arvaniti and M. Claassen. Sensitive detection of rare disease-associated cell subsets via representation learning.Nat Commun, 8:1–10, 2017
## Dependencies

* cellCNN makes use of golang (**go1.13.8**, the code was only tested with this golang version). For more information on how to install go follow this [link](https://golang.org/doc/install).
* All necessary libraries are automatically downloaded during the first execution ([go modules](https://blog.golang.org/using-go-modules)).
* cellCNN does an intensive use of [Overlay-network (ONet) library](https://github.com/dedis/onet) and of [Lattigo](https://github.com/ldsec/lattigo).
* For more information regarding the underlying architecture please refer to the stable version of ONet `go.dedis.ch/onet/v3`.
* cellCNN requires **Python3.7** and for data preprocessing and generation of the multi-cell inputs in `cellCNN/cellCNN_clear`. 
* cellCNN makes use of **conda 4.10.3** (the rest of the dependencies are defined under `enviroment.yml`).

## Overview
- `cellCNN/cellCNN_clear`: contains the relevant data preprocessing and distribution scripts and centralized and distributed cellCnn implementation for testing/benchmarking.
    - Data generation and preprocessing scripts require Python 3.6
- `cellCNN/cellCNN_optimized`: contains the optimized, distributed, privacy-preserving implementation of cellCnn with multiparty homomorphic encryption
- `cellCNN/cellcnnPoseidon`: contains the distributed, privacy-preserving implementation of cellCnn with multiparty homomorphic encryption, relying on POSEIDON[2] packing strategies, for benchmarking
- `cellCNN/eval_results`: contains the results for several experimental settings and the script for plotting accuracy, precision, recall, and f-score boxplots.
- `cellCNN/exampleData`: contains example data generated from CMV infection (NK dataset) for running an example of distributed and privacy-preserving implementation under `cellCNN/cellCNN_optimized/decentralized/example/`

[2] S. Sav, A. Pyrgelis, J. R. Troncoso-Pastoriza, D. Froelicher, J.-P. Bossuat,
  J. S. Sousa, and J.-P. Hubaux, “POSEIDON: Privacy-Preserving Federated
  Neural Network Learning,” in NDSS, 2021






  
