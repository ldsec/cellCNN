package cellCNN

import (
	"github.com/ldsec/lattigo/v2/ckks"
)



func EncryptRightForNaiveMul(W *Matrix, batchSize int, params ckks.Parameters, level int, encoder ckks.Encoder, encryptor ckks.Encryptor) (*ckks.Ciphertext){
	// Extract each column of W
	Wt := W.Transpose()

	values := make([]complex128, params.Slots())

	rows := Wt.Rows
	cols := Wt.Cols
	for i := 0; i < rows; i++{
		for j := 0; j < batchSize; j++{
			idx := i*Wt.Cols*batchSize
			copy(values[idx + j*cols: idx + (j+1)*cols], Wt.M[i*cols:(i+1)*cols])
		}
	}

	ptW := ckks.NewPlaintext(params, level, params.Scale())
	encoder.EncodeNTT(ptW, values, params.LogSlots())
	ctW := encryptor.EncryptNew(ptW)

	return ctW
}