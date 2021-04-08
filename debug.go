package cellCNN

import(
	"github.com/ldsec/lattigo/v2/ckks"
	"fmt"
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

func DecryptPrintMatrix(rows, cols int, Real bool, ciphertext *ckks.Ciphertext, params *ckks.Parameters, sk *ckks.SecretKey) {

	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	v := encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())

	fmt.Println(ciphertext.Level(), ciphertext.Scale())

	fmt.Printf("[\n")
	for i := 0; i < rows; i++ {
		fmt.Printf("[ ")
		for j := 0; j < cols; j++ {
			if Real {
				fmt.Printf("%11.8f, ", real(v[i*cols+j]))
			}else{
				fmt.Printf("%11.8f, ", v[i*cols+j])
			}
		}
		fmt.Printf("],\n")
	}
	fmt.Printf("]\n")
}