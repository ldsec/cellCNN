# Party Creation

P = cellCNN.NewCellCNNProtocol(params \*ckks.Parameters)

#Key Generation

Secret-Key : P.SetSecretKey(sk \*ckks.SecretKey)

Public-Key : p.SetPublicKey(pk \*ckks.PublicKey)

Relinearization-Key : P.SetRelinKey(sk \*ckks.RelinearizationKey)

Rotation-Key : P.SetRotationkey(galEl uint64, TOBEDEFINED)

To get the needed rotations : P.RotKeyIndex()

#Weight Init

// Roots 
C := cellCNN.WeightsInit(Features, Filters, Features)
W := cellCNN.WeightsInit(Filters, Classes (labels), Filters) 

// Then send C and W down to childrends


// Root and each children do
P.SetWeights(C \*ckks.Matrix, W \*ckks.Matrix)
P.EncryptWeights()


#Init Local Evaluator

// Root and all children do
P.EvaluatorInit()


# Local Computation

One epoch : 
1) Select batch of samples
2) Average prepooling of the samples
3) Forward pass
4) Bootstrapping with repacking
5) Backward pass
6) Global update

## 2) Pre-pooling
Load batch of n samples, each of Features x Filters, average pooling across the Features. Batch is now an n x Filters matrix

## 3) Forward Pass

### Plain
P.ForwardPlain(batch \*ckks.Matrix)

### Encrypted
P.Forward(batch \*ckks.Matrix)


## 4) Bootstrapping


## 5) Backward

### Plain

P.BackwardPlain(XBatch, YBatch \*ckks.Matrix, #Parties int))

### Encrypted

P.Backward(XBatch, YBatch \*ckks.Matrix, #Parties int)


## 6) Update Model

### Plain

P.UpdatePlain(DCPool, DWPool)

### Encrypted

P.Update(ctDCPool, ctDWPool \*ckks.Ciphertext,)