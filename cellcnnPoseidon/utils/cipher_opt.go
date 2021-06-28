package utils

import (
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
)

// AddHoistedMap add the hoisted ciphertext together
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

// MaskAndCollectToLeft is a naive implementation of collect slots to left
// comsume one level
// output ct without garbage slots
func MaskAndCollectToLeft(
	ct *ckks.Ciphertext, params ckks.Parameters,
	encoder ckks.Encoder, eval ckks.Evaluator,
	start, step, num int,
) *ckks.Ciphertext {
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
		if err := eval.Rescale(tmp, params.Scale(), tmp); err != nil {
			panic("fail to rescale, utils.MaskAndCollectToLeft")
		}
		if i == 0 {
			ctCollect = tmp
		} else {
			eval.Add(ctCollect, tmp, ctCollect)
		}
	}

	return ctCollect
}

// MaskAndCollectToLeftFast is a fast implementation of collect slots to left
// input ct with garbage slots
// output ct with garbage slots
// this funtion comsume (isMasked+outMask) level
// outMask: if mask garbage slots in output
func MaskAndCollectToLeftFast(
	ct *ckks.Ciphertext, params ckks.Parameters,
	encoder ckks.Encoder, eval ckks.Evaluator,
	start, step, num int, inMasked bool, outMask bool,
) *ckks.Ciphertext {
	if num > step-1 {
		panic("cannot fast collect, num > step-1")
	}
	var mct *ckks.Ciphertext
	if !inMasked {
		// fist mask the slots I want
		maskInds := NewSlice(0, step*(num-1), step)
		mask := GenSliceWithOneAt(params.Slots(), maskInds)
		maskPlain := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), mask, params.LogSlots())
		mct = eval.MulRelinNew(ct, maskPlain)
		if err := eval.Rescale(mct, params.Scale(), mct); err != nil {
			panic("fail to rescale, utils.MaskAndCollectToLeftFast")
		}
	} else {
		mct = ct.CopyNew()
	}
	// conduct innerSum to collect to left most with garbage slots
	eval.InnerSumLog(mct, step-1, num, mct)
	if outMask {
		outMask := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), GenSliceWithOneAt(params.Slots(), NewSlice(0, num-1, 1)), params.LogSlots())
		mct = eval.MulRelinNew(mct, outMask)
		if err := eval.Rescale(mct, params.Scale(), mct); err != nil {
			panic("fail to rescale, utils.MaskAndCollectToLeftFast")
		}
	}
	return mct
}

// DummyBootstrapping re-encrypt a ciphertext
func DummyBootstrapping(ct *ckks.Ciphertext, params ckks.Parameters, sk *rlwe.SecretKey) *ckks.Ciphertext {
	ect := ckks.NewEncryptorFromSk(params, sk)
	dct := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)
	cmplSlice := encoder.Decode(dct.DecryptNew(ct), params.LogSlots())
	replain := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), cmplSlice, params.LogSlots())
	return ect.EncryptNew(replain)
}
