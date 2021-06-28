package cellCNN

import (
	"fmt"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"math"
)

func DecryptPrint(rows, cols int, Real bool, element interface{}, params ckks.Parameters, sk *rlwe.SecretKey) {

	var v []complex128

	switch element := element.(type) {
	case *ckks.Ciphertext:
		decryptor := ckks.NewDecryptor(params, sk)
		encoder := ckks.NewEncoder(params)
		fmt.Println(element.Level(), element.Scale())
		v = encoder.Decode(decryptor.DecryptNew(element), params.LogSlots())

	case *ckks.Plaintext:
		encoder := ckks.NewEncoder(params)
		fmt.Println(element.Level(), element.Scale())
		v = encoder.Decode(element, params.LogSlots())
	}

	fmt.Printf("[\n")
	for i := 0; i < rows; i++ {
		fmt.Printf("[ ")
		for j := 0; j < cols; j++ {
			if Real {
				fmt.Printf("%11.8f, ", real(v[i*cols+j]))
			} else {
				fmt.Printf("%11.8f, ", v[i*cols+j])
			}
		}
		fmt.Printf("],\n")
	}
	fmt.Printf("]\n")
	fmt.Println()
}

func DecryptPrintMatrix(M *Matrix, Real bool, ciphertext *ckks.Ciphertext, params ckks.Parameters, sk *rlwe.SecretKey) {

	rows := M.Rows
	cols := M.Cols

	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	v := encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())

	M.Print()

	fmt.Println(ciphertext.Level(), ciphertext.Scale())

	sum := 0.0
	fmt.Printf("[\n")
	for i := 0; i < rows; i++ {
		fmt.Printf("[ ")
		for j := 0; j < cols; j++ {
			if Real {
				fmt.Printf("%11.8f, ", real(v[i*cols+j]))
				sum += math.Abs(real(M.M[i*cols+j]) - real(v[i*cols+j]))
			} else {
				fmt.Printf("%11.8f, ", v[i*cols+j])
			}
		}
		fmt.Printf("],\n")
	}
	fmt.Printf("]\n")

	fmt.Printf("Precision : %f\n", math.Log2(1/(sum/float64(rows*cols))))
	fmt.Println()
}
