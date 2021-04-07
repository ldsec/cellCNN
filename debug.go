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