package utils

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/ldsec/lattigo/v2/ckks"
)

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

func PrintCipherLevel(cipher *ckks.Ciphertext, params ckks.Parameters) string {
	return fmt.Sprintf("Level: %d (logQ = %d), Scale: 2^%f\n", cipher.Level(), params.LogQLvl(cipher.Level()), math.Log2(cipher.Scale()))
}

func PrintDebug(params ckks.Parameters, ciphertext *ckks.Ciphertext, valuesWant []complex128, decryptor ckks.Decryptor, encoder ckks.Encoder) (valuesTest []complex128) {

	valuesTest = encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())

	fmt.Println()
	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
	fmt.Printf("Scale: 2^%f\n", math.Log2(ciphertext.Scale()))
	fmt.Printf("Activation Test: %v ...\n", valuesTest[0:4])
	fmt.Printf("Activation Want: %v ...\n", valuesWant[0:4])
	fmt.Println()

	// precStats := ckks.GetPrecisionStats(params, nil, nil, valuesWant, valuesTest)

	// fmt.Println(precStats.String())

	return
}

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

func NegativeSlice(slice []int) []int {
	newSlice := make([]int, len(slice))
	for i := range slice {
		newSlice[i] = -slice[i]
	}
	return newSlice
}

func f(x complex128) complex128 {
	return 1 / (cmplx.Exp(-x) + 1)
}

func ClearRotInds(inds []int, mod int) []int {
	indSet := make(map[int]bool)
	for _, id := range inds {
		tid := ((id % mod) + mod) % mod
		if _, ok := indSet[tid]; !ok {
			indSet[tid] = true
		}
	}
	res := make([]int, 0)
	for k, _ := range indSet {
		res = append(res, k)
	}
	return res
}

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

func CopyCiphertextSlice(input []*ckks.Ciphertext) []*ckks.Ciphertext {
	res := make([]*ckks.Ciphertext, len(input))
	for i, each := range input {
		res[i] = each.CopyNew()
	}
	return res
}
