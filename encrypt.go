package cellCNN

import (
	"github.com/ldsec/lattigo/v2/ckks"
)

func EncryptRightForPtMul(C *ckks.Matrix, rowsA int, params *ckks.Parameters, level int, sk *ckks.SecretKey) (*ckks.Ciphertext){

	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptorFromSk(params, sk)

	rowsB := C.Rows()
	colsB := C.Cols()

	// Replicates x times the right matrix to accomodate
	// for its left rotations during the convolution
	replicate := int(float64(rowsA)/float64(rowsB) + 1.5)

	dxd := rowsB * colsB

	values := make([]complex128, params.Slots())
	for i := 0; i < replicate; i++ {
		for k, c := range C.M {
			values[i*dxd+k] = c
		}	
	}

	// If the left matrix is long and right matrix tall
	// then replicate = 1, and we must add (rowsA-1) * colsB
	// values of the matrix B
	if replicate == 1 {
		for k := 0; k < (rowsA-1) * colsB; k++ {
			values[replicate + k] = C.M[k]
		}
	}
	
	ptC := ckks.NewPlaintext(params, level, params.Scale())
	encoder.EncodeNTT(ptC, values, params.LogSlots())
	ctC := encryptor.EncryptNew(ptC)

	return ctC
}

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