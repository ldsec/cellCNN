package cellCNN

import (
	"github.com/ldsec/lattigo/v2/ckks"
)

// EncryptForPrediction takes as input a batch of samples (one sample per matrix) and returns the encryption of those samples
// formated for the forward pass into #Filters ciphertext.
// Maximum batch size (samples) per ciphertext : Slots / (Filters * Labels)
func EncryptForPredictionNew(XBatch []*Matrix, encoder ckks.Encoder, encryptor ckks.Encryptor, params ckks.Parameters) []*ckks.Ciphertext {

	XPrePool := new(Matrix)
	XBatchPrePool := NewMatrix(len(XBatch), Features)

	for i := range XBatch {
		XPrePool.SumColumns(XBatch[i])
		XPrePool.MultConst(XPrePool, complex(1.0/float64(Cells), 0))
		XBatchPrePool.SetRow(i, XPrePool.M)
	}

	ciphertexts := make([]*ckks.Ciphertext, Features>>1)
	for i := range ciphertexts {
		ciphertexts[i] = ckks.NewCiphertext(params, 1, 4, params.Scale())
	}

	EncryptLeftForCtMul(XBatchPrePool, Filters, 0.5, ciphertexts, encoder, encryptor, params)

	return ciphertexts
}

func EncryptForPrediction(XBatch []*Matrix, encoder ckks.Encoder, encryptor ckks.Encryptor, params ckks.Parameters, ciphertexts []*ckks.Ciphertext)  {

	XPrePool := new(Matrix)
	XBatchPrePool := NewMatrix(len(XBatch), Features)

	for i := range XBatch {
		XPrePool.SumColumns(XBatch[i])
		XPrePool.MultConst(XPrePool, complex(1.0/float64(Cells), 0))
		XBatchPrePool.SetRow(i, XPrePool.M)
	}

	EncryptLeftForCtMul(XBatchPrePool, Filters, 0.5, ciphertexts, encoder, encryptor, params)
}

// Predict applies the forward pass on a batch of encypted samples.
func Predict(XBatch []*ckks.Ciphertext, ctC, ctW *ckks.Ciphertext, params ckks.Parameters, eval ckks.Evaluator) (ctPredict *ckks.Ciphertext) {

	// Convolution
	ctP := MulMatrixLeftCtWithRightCt(XBatch, ctC, Features, Filters, eval)

	// Replicates the values for all the classes
	eval.Replicate(ctP, BatchSize*Filters, Classes, ctP)

	// Dense Layer
	ctU := DenseLayer(ctP, ctW, Filters, Classes, eval)

	eval.Add(ctU, eval.ConjugateNew(ctU), ctU)

	var err error
	if ctPredict, err = eval.EvaluatePoly(ctU, ckks.NewPoly(coeffsActivation), ctU.Scale()); err != nil {
		panic(err)
	}

	return
}
