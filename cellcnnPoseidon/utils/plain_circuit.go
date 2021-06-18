package utils

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/ldsec/lattigo/v2/ckks"
	"gonum.org/v1/gonum/mat"
)

func SliceCmplxToFloat64(arr []complex128) []float64 {
	res := make([]float64, len(arr))
	for i := range arr {
		res[i] = real(arr[i])
	}
	return res
}

func SliceFloat64ToCmplx(arr []float64) []complex128 {
	res := make([]complex128, len(arr))
	for i := range arr {
		res[i] = complex(arr[i], 0)
	}
	return res
}

// left rotate arr k times, negative k indicate a right rotation
func SliceRotation(arr []complex128, k int) []complex128 {
	// rt := make([]complex128, len(arr))
	var left, right []complex128
	k = (k%len(arr) + len(arr)) % len(arr)
	if k == 0 {
		return arr
	}
	left = arr[k:]
	right = arr[:k]
	return append(left, right...)
}

func SliceSum(arr []complex128, length int) complex128 {
	res := complex(0, 0)
	for i, x := range arr {
		if i >= length {
			break
		}
		res += x
	}
	return res
}

func SlicePow(arr []float64, n float64) []float64 {
	res := make([]float64, len(arr))
	for i := range arr {
		res[i] = math.Pow(arr[i], n)
	}
	return res
}

func SliceAdd(left []complex128, right []complex128, isNegative bool) []complex128 {
	if len(left) != len(right) {
		panic(fmt.Sprintf(
			"SliceAdd: length should be the same, got left [%d], right [%d]\n", len(left), len(right),
		))
	}

	res := make([]complex128, len(left))
	for i := 0; i < len(left); i++ {
		if isNegative {
			res[i] = left[i] - right[i]
		} else {
			res[i] = left[i] + right[i]
		}
	}
	return res
}

func SliceMult(left []complex128, right []complex128) []complex128 {
	if len(left) != len(right) {
		panic(fmt.Sprintf(
			"SliceMult: length should be the same, got left [%d], right [%d]\n", len(left), len(right),
		))
	}

	res := make([]complex128, len(left))
	for i := 0; i < len(left); i++ {
		res[i] = left[i] * right[i]
	}

	return res
}

func SliceMultConst(arr []complex128, k float64) []complex128 {
	ck := complex(k, 0)
	res := make([]complex128, len(arr))
	for i := range arr {
		res[i] = arr[i] * ck
	}
	return res
}

// collect inds slots to left most, others 0
func SliceCollect(arr []complex128, inds []int) []complex128 {
	res := make([]complex128, len(arr))
	for i, ind := range inds {
		res[i] = arr[ind]
	}
	return res
}

func SliceTranspose(arr []complex128, c, r int, rowPacked bool) []complex128 {
	res := make([]complex128, len(arr))
	var rid, cid, newi int
	for i := 0; i < len(res); i++ {
		if i >= c*r {
			break
		}
		// 1. compute the normal ind of the element in matrix
		if rowPacked {
			rid = i / c
			cid = i % c
			newi = cid*r + rid
		} else {
			rid = i % r
			cid = i / r
			newi = rid*c + cid
		}
		res[newi] = arr[i]
	}
	return res
}

// replicate the first n elements k tims as:
// [a,b,c,d, ...] => [a,b,c,d,a,b,c,d, ...] others slots set to 0
func SliceReplicate(arr []complex128, n, k int) []complex128 {
	res := make([]complex128, len(arr))
	for i := 0; i < n; i++ {
		for j := 0; j < k; j++ {
			res[i+j*n] = arr[i]
		}
	}
	return res
}

// e.g. if batch == 3, then every 3 elements will be sumed
// the batch * (0~n-1) will be the result, others set to 0
func SliceInnerSum(arr []complex128, batch, n int) []complex128 {
	res := make([]complex128, len(arr))
	for i := 0; i < n; i++ {
		res[batch*i] = arr[batch*i]
		for j := 1; j < batch; j++ {
			res[batch*i] += arr[batch*i+j]
		}
	}
	return res
}

// evaluate poly
func SliceEvaluatePoly(arr []complex128, coeffs []complex128, n int) []complex128 {
	res := make([]complex128, len(arr))
	for i := 0; i < len(arr); i++ {
		tmp := 0.0
		for degree, coeff := range coeffs {
			tmp += real(coeff) * math.Pow(real(arr[i]), float64(degree))
		}
		res[i] = complex(tmp, 0)
	}
	return res
}

// e.g.: arr = [2,0,0,1,0,0,3,0,0...], inds=[0,3,6], k=2
// (2,1,3 each twice) => return [2,2,0,1,1,0,3,3,0,...]
func SliceExtend(arr []complex128, inds []int, k int) []complex128 {
	res := make([]complex128, len(arr))
	for _, i := range inds {
		for j := 0; j < k; j++ {
			res[i+j] = arr[i]
		}
	}
	return res
}

func SliceFill(length int, inds []int, num complex128) []complex128 {
	res := make([]complex128, length)
	for _, i := range inds {
		res[i] = num
	}
	return res
}

func PlaintextLoss(pred []complex128, labels []float64) []complex128 {
	// use naive least square loss: L = \sum (pred_i - label_i)^2
	// d-L / d-pred_i = 2 (pred_i - label_i)

	// 1. prepare the plaintext labels
	clabels := make([]complex128, len(pred))
	for i := range clabels {
		if i >= len(labels) {
			break
		}
		clabels[i] = complex(labels[i], 0)
	}

	out1 := SliceAdd(pred, clabels, true)

	out2 := SliceMultConst(out1, 2)

	return out2
}

func DebugWithPlain(
	params ckks.Parameters, ciphertext *ckks.Ciphertext, valuesWant []complex128,
	decryptor ckks.Decryptor, encoder ckks.Encoder, inds []int,
) {

	valuesTest := encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())

	testReal := "Test Real:"
	testImg := "Test Img"
	wantReal := "Want Real"
	for _, i := range inds {
		testReal = fmt.Sprintf("%v, %v", testReal, real(valuesTest[i]))
		testImg = fmt.Sprintf("%v, %v", testImg, imag(valuesTest[i]))
		wantReal = fmt.Sprintf("%v, %v", wantReal, real(valuesWant[i]))
	}

	fmt.Println()
	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
	fmt.Printf("Scale: 2^%f\n", math.Log2(ciphertext.Scale()))
	fmt.Printf("Ciphertext Circuit:\n Real: %v\n Imaginary: %v\n", testReal, testImg)
	fmt.Printf("Plaintext Circuit:\n Real: %v\n", wantReal)
	fmt.Println()
	fmt.Printf("=> first 10 Ciphertext Circuit: %v\n", valuesTest[:10])
	fmt.Printf("=> first 10 Plaintext Circuit: %v\n\n", valuesWant[:10])
}

func GetRow(arr *mat.Dense, ind int) []float64 {
	_, c := arr.Dims()
	res := make([]float64, c)
	for i := 0; i < c; i++ {
		res[i] = arr.At(ind, i)
	}
	return res
}

func GetCol(arr *mat.Dense, ind int) []float64 {
	r, _ := arr.Dims()
	res := make([]float64, r)
	for i := 0; i < r; i++ {
		res[i] = arr.At(i, ind)
	}
	return res
}

func DebugWithDense(
	params ckks.Parameters, ciphertext *ckks.Ciphertext, valuesWant *mat.Dense,
	decryptor ckks.Decryptor, encoder ckks.Encoder, firstN int, inds []int, isRow bool,
) {
	// r, c := valuesWant.Dims()
	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
	fmt.Printf("Scale: 2^%f\n", math.Log2(ciphertext.Scale()))
	if isRow {
		fmt.Println("--> Rows of values want:")
		for _, i := range inds {
			fmt.Printf("   Row{%v}: %v\n", i, GetRow(valuesWant, i))
		}
	} else {
		fmt.Println("--> Cols of values want:")
		for _, i := range inds {
			fmt.Printf("   Col{%v}: %v\n", i, GetCol(valuesWant, i))
		}
	}
	valuesTest := encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())
	fmt.Println("--> Values test:")
	fmt.Printf("    First{%v}: %v\n", firstN, valuesTest[:firstN])
}

func ComputeDimWithIndex(i, r, c int, rowPacked bool) (int, int) {
	var dim0, dim1 int
	if rowPacked {
		dim0 = i / c
		dim1 = i % c
	} else {
		dim0 = i % r
		dim1 = i / r
	}
	return dim0, dim1
}

// return the mean square error between the plaintext dense and the ciphertext
func DebugCtWithDenseStatistic(
	params ckks.Parameters, ciphertext *ckks.Ciphertext, valuesWant *mat.Dense,
	decryptor ckks.Decryptor, encoder ckks.Encoder, rowPacked bool, verbose bool,
) float64 {
	r, c := valuesWant.Dims()
	valuesTest := encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())
	var dim0, dim1 int

	mse := 0.0

	for i := 0; i < r*c; i++ {
		dim0, dim1 = ComputeDimWithIndex(i, r, c, rowPacked)
		target := valuesWant.At(dim0, dim1)
		pred := real(valuesTest[i])
		mse += math.Pow(target-pred, 2.0)
	}

	prob := rand.Intn(r * c)
	dim0, dim1 = ComputeDimWithIndex(prob, r, c, rowPacked)
	mse = mse / float64(r*c)

	if verbose {
		fmt.Printf(">> Debug with plaintext | index: %v, want: %v, get: %v, MSE: %v\n",
			prob, valuesWant.At(dim0, dim1), real(valuesTest[prob]), mse)
	}

	return mse
}

// return the mean square error between the plaintext dense and the ciphertext
func DebugCtSliceWithDenseStatistic(
	params ckks.Parameters, ciphertexts []*ckks.Ciphertext, valuesWant *mat.Dense,
	decryptor ckks.Decryptor, encoder ckks.Encoder, rowPacked bool, verbose bool,
) float64 {
	r, c := valuesWant.Dims()
	valuesTest := make([][]complex128, len(ciphertexts))
	for i, each := range ciphertexts {
		valuesTest[i] = encoder.Decode(decryptor.DecryptNew(each), params.LogSlots())
	}
	var dim0, dim1 int
	var pred float64

	mse := 0.0

	for i := 0; i < r*c; i++ {
		dim0, dim1 = ComputeDimWithIndex(i, r, c, rowPacked)
		target := valuesWant.At(dim0, dim1)
		if rowPacked {
			pred = real(valuesTest[dim0][dim1])
		} else {
			pred = real(valuesTest[dim1][dim0])
		}
		mse += math.Pow(target-pred, 2.0)
	}

	mse = mse / float64(r*c)

	prob := rand.Intn(r * c)
	dim0, dim1 = ComputeDimWithIndex(prob, r, c, rowPacked)
	if rowPacked {
		pred = real(valuesTest[dim0][dim1])
	} else {
		pred = real(valuesTest[dim1][dim0])
	}

	if verbose {
		fmt.Printf(">> Debug with plaintext | index: %v, want: %v, get: %v, MSE: %v\n",
			prob, valuesWant.At(dim0, dim1), pred, mse)
	}

	return mse
}
