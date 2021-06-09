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

// collect the desired slots to left most
// this function consume one level
// e.g.
// input:
// 		unmasked ciphertext: [a, garb, garb, garb, b, garb, garb, garb, c, garb, garb, garb, ...]
// result:
//  	ciphertext: [a, b, c, garb, garb, garb...]
//		(actually: [a, b, c, 0, b, c, 0, 0, c, 0, 0, 0,...])
func MaskAndCollectToLeft(
	ct *ckks.Ciphertext, params ckks.Parameters,
	encoder ckks.Encoder, eval ckks.Evaluator,
	start, step, num int,
) *ckks.Ciphertext {
	// +++++++++++++++++++++++++++++
	// if num > step-1 {
	// 	panic("cannot fast collect, num > step-1")
	// }
	// // fist mask the slots I want
	// maskInds := NewSlice(0, step*(num-1), step)
	// mask := GenSliceWithOneAt(params.Slots(), maskInds)
	// maskPlain := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), mask, params.LogSlots())
	// mct := eval.MulRelinNew(ct, maskPlain)
	// // conduct innerSum to collect to left most with garbage slots
	// eval.InnerSum(mct, step-1, num, mct)
	// return mct
	// +++++++++++++++++++++++++++++

	// -----------------------------
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
	// -----------------------------
}

func MaskAndCollectToLeftFast(
	ct *ckks.Ciphertext, params ckks.Parameters,
	encoder ckks.Encoder, eval ckks.Evaluator,
	start, step, num int,
) *ckks.Ciphertext {
	// +++++++++++++++++++++++++++++
	if num > step-1 {
		panic("cannot fast collect, num > step-1")
	}
	// fist mask the slots I want
	maskInds := NewSlice(0, step*(num-1), step)
	mask := GenSliceWithOneAt(params.Slots(), maskInds)
	maskPlain := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), mask, params.LogSlots())
	mct := eval.MulRelinNew(ct, maskPlain)
	if err := eval.Rescale(mct, params.Scale(), mct); err != nil {
		panic("fail to rescale, utils.MaskAndCollectToLeftFast")
	}
	// conduct innerSum to collect to left most with garbage slots
	eval.InnerSumLog(mct, step-1, num, mct)
	return mct
}
