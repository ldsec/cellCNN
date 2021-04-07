package cellCNN

import (
	"github.com/ldsec/lattigo/v2/ckks"
)


func EncodePlaintextForE1(Y *ckks.Matrix, features, filters, classes int, params *ckks.Parameters) (*ckks.Plaintext){

	encoder := ckks.NewEncoder(params)

	values := make([]complex128, params.Slots())

	for i := 0; i < classes; i++ {
		c := Y.M[i]
		for j := 0; j < filters; j++ {
			values[i*filters + j] = c
		}
	}

	idx := classes * filters

	for i := 0; i < classes; i++ {
		c := Y.M[i]
		for j := 0; j < filters*features; j++ {
			values[idx + i*filters*features + j] = c
		}
	}

	pt := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	encoder.EncodeNTT(pt, values, params.LogSlots())

	return pt
}