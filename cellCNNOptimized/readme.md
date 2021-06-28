# CellCNN Optimized

## Parameters

The ML and crypto parameters can be set throuh the file `params.go`.

The following inequality must be respected to ensure the ciphertext has enough available slots to store all the data :

_3onh+(2o+1)(nh+(m/2−1)2h+h)≤N/2_

for _o_ the number of `Classes` (labels), _h_ the number of `Features`, _m_ the number of `Filters` (markers) and _n_ the `BatchSize`. In this implementation, the number of `Cells` per `Sample` does not impact the slot usage or training time.

The value `LogN` can be increase to accomodate for more slots.

The method `GenParams()` will automatically generate a `ckks.Parameters` with secure parameters for the circuit. 

## Testing

The file `decentralized/example/main.go` is an example of decentralized training (the centralized training can be emulated by setting the number of hosts to 1).

## PREDICTION API

Maximum batch size (samples) per ciphertext : Slots / (Filters * Labels)

```Go
EncryptForPrediction(XBatch []*Matrix, encoder ckks.Encoder, encryptor ckks.Encryptor, params ckks.Parameters) ([]*ckks.Ciphertext)
```

```Go
Predict(XBatch []*ckks.Ciphertext, ctC, ctW *ckks.Ciphertext, params ckks.Parameters, eval ckks.Evaluator) (*ckks.Ciphertext)
```

## TRAINING API 

### Party Creation

```Go
P = cellCNN.NewCellCNNProtocol(params ckks.Parameters)
```

### Key Generation

Set the secret key

```Go
P.SetSecretKey(sk *rlwe.SecretKey)
```

Set the public key

```Go
P.SetPublicKey(pk *rlwe.PublicKey)
```

Get the rotations indexes for the rotation keys

```Go
- P.RotKeyIndex()
```

Init the evaluator with relinearization key and rotation keys

```Go
- P.EvaluatorInit(rlk *rlwe.RelinearizationKey, rtk *rlwe.RotationKeySet)
```

### Weight Init

1) Roots does : 

```Go
C := cellCNN.WeightsInit(Features, Filters, Features int) (*cellCNN.Matrix)
W := cellCNN.WeightsInit(Filters, Classes, Filters int) (*cellCNN.Matrix)
``` 

2) Roots sends ``C`` and ``W`` down and to all childrends

3) Root and each children do :

```Go
P.SetWeights(C, W *Matrix)
P.EncryptWeights()
```

### Local Computation

One epoch :

1) Load data 
2) Select local batch of samples and average prepool
3) Forward pass
4) Bootstrapping
5) Backward pass
6) Aggregate weights
7) Model Update

#### 1) Load Data

```Go
XTrain, YTrain := cellCNN.LoadTrainDataFrom(path string, Samples, cellCNN.Cells, cellCNN.Features int)
XValid, YValid := cellCNN.LoadValidDataFrom(path string, Samples, cellCNN.Cells, cellCNN.Features int)
```

#### 2) Select Batch of Samples and Average Prepool

```Go
XPrepool := new(cellCNN.Matrix)
XBatch := cellCNN.NewMatrix(cellCNN.BatchSize, cellCNN.Features int)
YBatch := cellCNN.NewMatrix(cellCNN.BatchSize, cellCNN.Classes int)

// Loads a batch of samples
for k := 0; k < cellCNN.BatchSize; k++ {

	randi := rand.Intn(partyDataSize int)

	X := XTrain[randi]
	Y := YTrain[randi]

	// Average pre-pooling
	XPrePool.SumColumns(X)
	XPrePool.MultConst(XPrePool, complex(1.0/float64(cellCNN.Cells), 0))

	XBatch.SetRow(k, XPrePool.M)
	YBatch.SetRow(k, []complex128{Y.M[1], Y.M[0]})
}
```

#### 3) Forward Pass

##### Plain
```Go
P.ForwardPlain(XBatch)
```

##### Encrypted
```Go
P.Forward(XBatch)
```

#### 4) Bootstrapping (dummy)
```Go
P.Refresh(masterSk *ckks.SecretKey, nParties int)
```

#### 5) Backward

##### Plain

```Go
P.BackwardPlain(XBatch, YBatch *cellCNN.Matrix, nParties int)
```
##### Encrypted

```Go
P.Backward(XBatch, YBatch *cellCNN.Matrix, nParties int)
```

#### 6) Aggregate Weights

##### Plain
```Go
P.DC.Add(P.DC, P'.DC)
P.DW.Add(P.DW, P'.DW)
```

##### Encrypted

Aggregates the weights of P' on the weights of P.

```Go
P.Eval.Add(P.CtDC, P'.CtDC, P.CtDC)
P.Eval.Add(P.CtDW, P'.CtDW, P.CtDW)
```

#### 7) Update Model

Updates the local model (with momentum) using the internal aggregated weights.

##### Plain

```Go
P.UpdatePlain()
```

##### Encrypted

```Go
P.Update()
```

### Debug

#### Updated Weights

```Go
fmt.Println("DC")
P.DC.Print()
cellCNN.DecryptPrint(cellCNN.Features, cellCNN.Filters, true, P.CtDC, params, masterSk)

fmt.Println("DW")
P.DW.Transpose().Print()
for i := 0; i < cellCNN.Classes; i++{
	cellCNN.DecryptPrint(1, cellCNN.Filters, true, P.Eval().RotateNew(P.CtDW, i*cellCNN.BatchSize*cellCNN.Filters), params, masterSk)
}
```

#### Aggregated Weights

```Go
fmt.Println("DCPool")
P.C.Print()
cellCNN.DecryptPrint(cellCNN.Features, cellCNN.Filters, true, P.CtC, params, masterSk)

fmt.Println("DWPool")
P.W.Transpose().Print()
for i := 0; i < cellCNN.Classes; i++{
	cellCNN.DecryptPrint(1, cellCNN.Filters, true, P.Eval().RotateNew(P.CtW, i*cellCNN.BatchSize*cellCNN.Filters), params, masterSk)
}
```

#### Weights

```Go
P.PrintCtWPrecision(masterSk)
P.PrintCtCPrecision(masterSk)
```