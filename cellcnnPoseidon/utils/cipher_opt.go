package utils

import (
	"github.com/ldsec/lattigo/v2/ckks"
)

func AddHoistedMap(hm map[int]*ckks.Ciphertext, eval ckks.Evaluator) *ckks.Ciphertext {
	var res *ckks.Ciphertext
	for _, v := range hm {
		if res == nil {
			res = v
		} else {
			res = eval.AddNew(res, v)
		}
	}
	return res
}

// consume one level
func MaskAndCollectToLeft(
	ct *ckks.Ciphertext, params *ckks.Parameters,
	encoder ckks.Encoder, eval ckks.Evaluator,
	start, step, num int,
) *ckks.Ciphertext {
	// first rotate hoisted to put the slots into the place I want:
	rotInds := make([]int, num)
	for i := 0; i < num; i++ {
		rotInds[i] = (start + step*i) - i
	}

	rotMap := eval.RotateHoisted(ct, rotInds)

	var ctCollect *ckks.Ciphertext
	for i := 0; i < num; i++ {
		// create mask
		mask := GenSliceWithOneAt(params.Slots(), []int{i})
		maskPlain := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), mask, params.LogSlots())

		// mult with mask plaintext
		tmp := eval.MulRelinNew(rotMap[(start+step*i)-i], maskPlain)
		if i == 0 {
			ctCollect = tmp
		} else {
			eval.Add(ctCollect, tmp, ctCollect)
		}
	}
	if err := eval.Rescale(ctCollect, params.Scale(), ctCollect); err != nil {
		panic("fail to rescale, utils.MaskAndCollectToLeft")
	}
	// fmt.Printf("MaskAndCollectToLeft " + PrintCipherLevel(ctCollect, params))

	return ctCollect
}
