# Semester Project of Shufan Wang

## Introduction

This is the Repository for Semester Project in Computer Science II (12 credits) of Shufan Wang at [Laboratory of Data Security (LDS)](https://lds.epfl.ch/), EPFL 2021.

## Current Settings

Currently we use a set of crypto params below:
```
params := ckks.ParametersLiteral{
		LogN:     15,
		LogSlots: 14,
		LogQ:     []int{60, 60, 60, 52, 52, 52, 52, 52, 52, 52}, // 544
		LogP:     []int{61, 61, 61},                             // 183
		Scale:    1 << 52,
		Sigma:    rlwe.DefaultSigma,
	}
```
The initial (or max) level of ciphertext is 9. We use 1 local bootstrapping in each participants during the forward and backward pass. And the aggregate gradients with momentum is at level 2. Then the aggregate server will use one bootstrapping to recover the gradients to level 9 for weights update.

## Project Structure

- `cellcnnPoseidon/layers`: define the ciphertext circuit of Conv1D layer and Linear layer.

- `cellcnnPoseidon/centralized`: define the Cell CNN struct with initialization, forward, and backward.

- `cellcnnPoseidon/decentralized`: define a privacy-preserving federate learning protocol where each participant holds local data and together train a global Cell CNN by gradients aggregation.

## How to Run

- To test the centralized forward and backward with a plaintext net, enter `centralized/` and run `$ go test -run TestWithPlainNetBwBatch`. It will init an encrypted net and a plaintext net, and compare the forward prediction and backward gradients.

- To test the decentralized protocol, enter `decentralized/` and run `$ go test -run TestDemo`. It will init a test according to the parameters in `decentralized/settings.go`.