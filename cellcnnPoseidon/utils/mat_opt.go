package utils

import "gonum.org/v1/gonum/mat"

func RetrieveCol(arr *mat.Dense, ind int) []complex128 {
	r, _ := arr.Dims()
	res := make([]complex128, r)
	for i := 0; i < r; i++ {
		res[i] = complex(arr.At(i, ind), 0)
	}
	return res
}

func RetrieveRow(arr *mat.Dense, ind int) []complex128 {
	_, c := arr.Dims()
	res := make([]complex128, c)
	for i := 0; i < c; i++ {
		res[i] = complex(arr.At(ind, i), 0)
	}
	return res
}
