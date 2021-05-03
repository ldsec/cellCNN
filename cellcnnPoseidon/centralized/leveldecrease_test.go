package centralized

import (
	"fmt"
	"testing"

	"github.com/ldsec/cellCNN/celcnnPoseidon/utils"
	"github.com/ldsec/lattigo/v2/ckks"
)

func TestLevel(t *testing.T) {
	params := ckks.DefaultParams[ckks.PN14QP438]
	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("         INSTANTIATING SCHEME            ")
	fmt.Println("=========================================")
	fmt.Println()

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk)
	encryptor := ckks.NewEncryptorFromSk(params, sk)
	// decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)
	eval := ckks.NewEvaluator(params, ckks.EvaluationKey{Rlk: rlk})

	// ncells := 2
	// nmakers := 4
	nfilters := 3
	nclasses := 2

	// use predefined weights, column packed 3*2
	slots := params.Slots()
	plainWeights := make([]complex128, slots)
	for i, _ := range plainWeights {
		if i >= nfilters*nclasses {
			break
		}
		plainWeights[i] = complex(float64(i%4), 0)
	}
	encodeWeights := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), plainWeights, params.LogSlots())
	encryptWeights := encryptor.EncryptNew(encodeWeights)

	// use predefined input, one row 1*3
	plainInput := make([]complex128, slots)
	for i, _ := range plainInput {
		if i >= nfilters {
			break
		}
		plainInput[i] = complex(float64(i), 0)
	}

	encodeInput := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), plainInput, params.LogSlots())
	encryptInput := encryptor.EncryptNew(encodeInput)

	fmt.Println("=> Initial operand:")
	utils.PrintCipherLevel(encryptWeights, params)

	fmt.Println("==> After Multiplication:")
	mt1 := eval.MulRelinNew(encryptWeights, encryptInput)
	if err := eval.Rescale(mt1, params.Scale(), mt1); err != nil {
		panic("fail to rescale")
	}
	utils.PrintCipherLevel(mt1, params)

	fmt.Println("==> After Masking:")
	mt2 := eval.MulRelinNew(encryptWeights, encodeInput)
	if err := eval.Rescale(mt2, params.Scale(), mt2); err != nil {
		panic("fail to rescale")
	}
	utils.PrintCipherLevel(mt2, params)
}
