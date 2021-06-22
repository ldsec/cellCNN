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
The initial (or max) level of ciphertext is 9. We use one local bootstrapping in each participants during the forward and backward pass. And the aggregate gradients with momentum is at level 2. Then the aggregate server will use one bootstrapping to recover the gradients to level 9 for weights update.

## Project Structure

- `cellcnnPoseidon/layers`: define the ciphertext circuit of Conv1D layer and Linear layer.

- `cellcnnPoseidon/centralized`: define the Cell CNN struct with initialization, forward, and backward.

- `cellcnnPoseidon/decentralized`: define a privacy-preserving federate learning protocol where each participant holds local data and together train a global Cell CNN by gradients aggregation.

## How to Use

To init a local Cell CNN:
```
model := NewCellCNN(cnnSettings, cryptoParams, momentum, lr)
cw, dw := model.InitWeights(nil, nil, nil, nil) // random init weights, output the cleartext weights in matrix for plainnet
model.InitEvaluator(cryptoParams, maxM1N2Ratio) // init rotation keys, diagM...
model.sk = sk // for dummy bootstrapping
```

For batch forward and backward, please refer to `cellcnnPoseidon/benchmark_test.go`

## Quick Test
- To test the centralized forward and backward with a plaintext net, enter `centralized/` and run `$ go test -run TestWithPlainNetBwBatch`. It will init an encrypted net and a plaintext net, and compare the forward prediction and backward gradients.

- To test the decentralized protocol, enter `decentralized/` and run `$ go test -run TestDemo`. It will init a test according to the parameters in `decentralized/settings.go`.

## Conv1d Layer

The weights of the layer is ciphertext at level 9 (initial level).
If the plaintext weight matrix has 4 filters each with 2 makers like:
$$
W_{2\times4}=\left(\begin{array}{cc} 
a_1 &b_1 &c_1 &d_1\\
a_2 &b_2 &c_2 &d_2
\end{array}\right)
$$ 
The ciphertext weights slots will be pre-replicated for Ncells. If $N_{cells}=3$, it will be like:
$$
Filter1=\left(\begin{array}{cc} 
a_1 &a_2 &a_1 &a_2 &a_1 &a_2 &0.0 &...
\end{array}\right)\\
Filter2=\left(\begin{array}{cc} 
b_1 &b_2 &b_1 &b_2 &b_1 &b_2 &0.0 &...
\end{array}\right)\\
Filter3=\left(\begin{array}{cc} 
c_1 &c_2 &c_1 &c_2 &c_1 &c_2 &0.0 &...
\end{array}\right)\\
Filter4=\left(\begin{array}{cc} 
d_1 &d_2 &d_1 &d_2 &d_1 &d_2 &0.0 &...
\end{array}\right)\\
$$

**Conv1d Forward** Return the filter response average pooled over current batch cells. Following the settings above, $N_{cell}=3, N_{filter}=4, N_{maker}=2$, the forward ciphertext output are at the left most slots like:
$$
Activation=\left(\begin{array}{cc} 
a &b &c &d &0.0 &...
\end{array}\right)
$$

**Conv1d Backward** Return the gradients scaled with learning rate, without computing with momentum.


## Dense Layer

The weights of the layer is ciphertext at level 9 (initial level).
If the plaintext weight matrix has $N_{filter}=4, N_{class}=2$ like:
$$
W_{4\times2}=\left(\begin{array}{cc} 
e_1 &q_1\\
e_2 &q_2\\
e_3 &q_3\\
e_4 &q_4
\end{array}\right)
$$ 
The column packed ciphertext weights slots will be like $W_{dense}$, which will be multiplied with $Actv_{rep}$, the conv1d activation replicated by $N_{class}=2$ times.
$$
W_{dense}=\left(\begin{array}{cc} 
e_1 &e_2 &e_3 &e_4 &q_1 &q_2 &q_3 &q_4 &0.0 &...
\end{array}\right)\\
Actv_{rep}= \left(\begin{array}{cc} 
a &b &c &d &a &b &c &d &0.0 &...
\end{array}\right)
$$

**Dense Forward** output the prediction for each class, which will be at the indices of $i\cdot N_{filter}$ for i in $[0, N_{class})$ as:

$$
Pred=\left(\begin{array}{cc} 
p_1 &garb &garb &garb &p_2 &garb &garb &garb &0.0 &...
\end{array}\right)
$$

**Dense Backward** relates to one collective bootstrapping, which is replaced by a decrypt-re-encrypt function.


## Ciphertext Level Tracing

Local computation:
- Inital weights: 9
- Conv1d foward (activation): 7
- Dense forward (prediction): 4
- Dense backward: gradient (one bootstrapping used): 6, output err: 7
- Conv1d bacward: gradient: 4



  
