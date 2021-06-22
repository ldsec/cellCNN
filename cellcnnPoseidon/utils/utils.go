package utils

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/ldsec/lattigo/v2/ckks"
)

// PrintTime debug use only, the last one of tl shold be the sum
func PrintTime(tl []float64, i *int, label string) {
	var info string
	if len(tl) == 0 {
		panic("PrintTime get len(tl) == 0")
	}
	info = "==> Time: "
	if len(tl) == 1 {
		info = info + fmt.Sprintf("sum: %v (seconds)\n", tl[0])
	} else {
		for i, val := range tl {
			if i == len(tl)-1 {
				info = info + fmt.Sprintf("sum: %v (seconds)\n", val)
			} else {
				info = info + fmt.Sprintf("period %v: %v | ", i, val)
			}
		}
	}
	if i != nil {
		info = fmt.Sprintf("id: %v", *i) + info
	}
	info = label + " " + info
	fmt.Printf(info)
}

// PrintCipherLevel return the ciphertext level string
func PrintCipherLevel(cipher *ckks.Ciphertext, params ckks.Parameters) string {
	return fmt.Sprintf("Level: %d (logQ = %d)\n", cipher.Level(), params.LogQLvl(cipher.Level()))
}

// PrintDebug copied from lattigo test
func PrintDebug(params ckks.Parameters, ciphertext *ckks.Ciphertext, valuesWant []complex128, decryptor ckks.Decryptor, encoder ckks.Encoder) (valuesTest []complex128) {

	valuesTest = encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())

	fmt.Println()
	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
	fmt.Printf("Scale: 2^%f\n", math.Log2(ciphertext.Scale()))
	fmt.Printf("Activation Test: %v ...\n", valuesTest[0:4])
	fmt.Printf("Activation Want: %v ...\n", valuesWant[0:4])
	fmt.Println()

	return
}

// NewSlice init a new slice
func NewSlice(start, end, step int) []int {
	if step <= 0 || end < start {
		return []int{}
	}
	s := make([]int, 0, 1+(end-start)/step)
	for start <= end {
		s = append(s, start)
		start += step
	}
	return s
}

// NegativeSlice (a, b, c) => (-a, -b, -c)
func NegativeSlice(slice []int) []int {
	newSlice := make([]int, len(slice))
	for i := range slice {
		newSlice[i] = -slice[i]
	}
	return newSlice
}

// sigmoid function
func f(x complex128) complex128 {
	return 1 / (cmplx.Exp(-x) + 1)
}

// ClearRotInds remove the duplicate element in a slice with respect to a mod
func ClearRotInds(inds []int, mod int) []int {
	indSet := make(map[int]bool)
	for _, id := range inds {
		tid := ((id % mod) + mod) % mod
		if _, ok := indSet[tid]; !ok {
			indSet[tid] = true
		}
	}
	res := make([]int, 0)
	for k := range indSet {
		res = append(res, k)
	}
	return res
}

// CiphertextsToBytes transform a ciphertext slice to bytes
func CiphertextsToBytes(ct []*ckks.Ciphertext) [][]byte {
	data := make([][]byte, len(ct))
	var err error
	for i := 0; i < len(ct); i++ {
		data[i], err = ct[i].MarshalBinary()
		if err != nil {
			panic("fail to marshall ciphertext")
		}
	}
	return data
}

// CopyCiphertextSlice copy a ciphertext slice
func CopyCiphertextSlice(input []*ckks.Ciphertext) []*ckks.Ciphertext {
	res := make([]*ckks.Ciphertext, len(input))
	for i, each := range input {
		res[i] = each.CopyNew()
	}
	return res
}

// AVGandStdev compute the avg and stdev of a slice
func AVGandStdev(slice []float64) (float64, float64) {
	avg := 0.0
	std := 0.0

	for _, item := range slice {
		avg += item
	}

	avg = avg / float64(len(slice))

	for _, item := range slice {
		std += math.Pow(item-avg, 2.0)
	}
	std = std / float64(len(slice))
	std = math.Pow(std, 0.5)
	return avg, std
}
