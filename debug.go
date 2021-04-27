package cellCNN

import(
	"github.com/ldsec/lattigo/v2/ckks"
	"fmt"
	"math"
)

func DecryptPrint(start, finish int, ciphertext *ckks.Ciphertext, params *ckks.Parameters, sk *ckks.SecretKey) {

	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	v := encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())

	fmt.Println(ciphertext.Level(), ciphertext.Scale())
	for i := start; i < finish; i++ {
		fmt.Println(i, v[i])
	}
	fmt.Println()
}

func DecryptPrintMatrix(M *ckks.Matrix, Real bool, ciphertext *ckks.Ciphertext, params *ckks.Parameters, sk *ckks.SecretKey) {

	rows := M.Rows()
	cols := M.Cols()

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
			}else{
				fmt.Printf("%11.8f, ", v[i*cols+j])
			}
		}
		fmt.Printf("],\n")
	}
	fmt.Printf("]\n")

	fmt.Printf("Precision : %f\n", math.Log2(1/(sum / float64(rows*cols))))
	fmt.Println()
}