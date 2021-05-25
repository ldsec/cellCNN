package cellCNN

import (
	"github.com/ldsec/lattigo/v2/ckks"
)



func EncryptRightForNaiveMul(W *ckks.Matrix, batchSize int, params *ckks.Parameters, level int, encoder ckks.Encoder, encryptor ckks.Encryptor) (*ckks.Ciphertext){
	// Extract each column of W
	Wt := W.Transpose()

	values := make([]complex128, params.Slots())

	for i := 0; i < Wt.Rows(); i++{
		for j := 0; j < batchSize; j++{
			idx := i*Wt.Cols()*batchSize
			copy(values[idx + j*Wt.Cols(): idx + (j+1)*Wt.Cols()], Wt.M[i*Wt.Cols():(i+1)*Wt.Cols()])
		}
	}

	ptW := ckks.NewPlaintext(params, level, params.Scale())
	encoder.EncodeNTT(ptW, values, params.LogSlots())
	ctW := encryptor.EncryptNew(ptW)

	return ctW
}