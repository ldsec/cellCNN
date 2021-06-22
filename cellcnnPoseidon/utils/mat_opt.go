package utils

import (
	"math/rand"

	"github.com/ldsec/lattigo/v2/ckks"
	"gonum.org/v1/gonum/mat"
)

// GenRandomMatrix Generate a r×c matrix of random values.
func GenRandomMatrix(r, c int) *mat.Dense {
	data := make([]float64, r*c)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	return mat.NewDense(r, c, data)
}

// RetrieveCol return a col of a matrix
func RetrieveCol(arr *mat.Dense, ind int) []complex128 {
	r, _ := arr.Dims()
	res := make([]complex128, r)
	for i := 0; i < r; i++ {
		res[i] = complex(arr.At(i, ind), 0)
	}
	return res
}

// RetrieveRow return a row of a matrix
func RetrieveRow(arr *mat.Dense, ind int) []complex128 {
	_, c := arr.Dims()
	res := make([]complex128, c)
	for i := 0; i < c; i++ {
		res[i] = complex(arr.At(ind, i), 0)
	}
	return res
}

// Batch2PlainSlice transform a batch of matrices to plaintexts
func Batch2PlainSlice(inputs []*mat.Dense, params ckks.Parameters, encoder ckks.Encoder) []*ckks.Plaintext {
	result := make([]*ckks.Plaintext, 0)
	for _, each := range inputs {
		result = append(result, Matrix2Plaintext(each, params, encoder))
	}
	return result
}

// Matrix2Plaintext transform a matrix to plaintext
func Matrix2Plaintext(rawData *mat.Dense, params ckks.Parameters, encoder ckks.Encoder) *ckks.Plaintext {
	// shape of each input: 200 * 37 (ncells = 200, nmakers = 37)
	row, col := rawData.Dims()
	// fmt.Println("row, col: ", row, col)
	value := make([]complex128, params.Slots())
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			value[i*col+j] = complex(rawData.At(i, j), 0)
		}
	}

	return encoder.EncodeNTTAtLvlNew(params.MaxLevel(), value, params.LogSlots())
}
