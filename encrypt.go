package cellCNN

import (
	"github.com/ldsec/lattigo/v2/ckks"
)



func EncryptRightForNaiveMul(W *ckks.Matrix, params *ckks.Parameters, level int, sk* ckks.SecretKey) (*ckks.Ciphertext){

	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptorFromSk(params, sk)

	// Extract each column of W
	Wt := W.Transpose()

	values := make([]complex128, params.Slots())

	for k, c := range Wt.M {
		values[k] = c
	}	

	ptW := ckks.NewPlaintext(params, level, params.Scale())
	encoder.EncodeNTT(ptW, values, params.LogSlots())
	ctW := encryptor.EncryptNew(ptW)

	return ctW
}