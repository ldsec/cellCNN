# TODO 

1) Replace ckks.Matrix by go native package or local matrix package
2) Provide API to set keys to the protocol
3) Provide API to access internal ciphertexts
4) Provide API for bootstrapping (independant of the protocol)
5) Add pools for aggregation

# Party Creation

P = cellCNN.NewCellCNNProtocol(params \*ckks.Parameters)

# Key Generation

## Secret Key 

P.SetSecretKey(sk \*ckks.SecretKey)

## Encryption Key 

p.SetPublicKey(pk \*ckks.PublicKey)

## Relinearization Key 

P.SetRelinKey(sk \*ckks.RelinearizationKey)

## Rotation-Key 

To get the needed rotations : P.RotKeyIndex()

P.SetRotationkey(galEl uint64, TOBEDEFINED)


# Weight Init

1) Roots does : 

C := cellCNN.WeightsInit(Features, Filters, Features)

W := cellCNN.WeightsInit(Filters, Classes (labels), Filters) 

2) Roots sends C and W down and to all childrends

3) Root and each children do :

P.SetWeights(C \*ckks.Matrix, W \*ckks.Matrix)

then 

P.EncryptWeights()


# Init Local Evaluator

1) Root and all children do
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