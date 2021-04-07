package cellCNN

import(
	"github.com/ldsec/lattigo/v2/ckks"
)

func Backward(ctBoot *ckks.Ciphertext, Y *ckks.Plaintext, cells, features, filters, classes int, params *ckks.Parameters, eval ckks.Evaluator, sk *ckks.SecretKey) (ctDW, ctDC *ckks.Ciphertext){


	// sigma(U) and sigma'(U)
	ctU1, ctU1Deriv := ActivationsCt(ctBoot, params.Scale(), eval)

	// Y - sigma(U)
	eval.Sub(Y, ctU1, ctU1)

	// E1 = sigma'(U) * (Y - sigma(U))
	eval.MulRelin(ctU1, ctU1Deriv, ctU1)
	if err := eval.Rescale(ctU1, params.Scale(), ctU1); err != nil{
		panic(err)
	}

	// Ppool * E1 
	ctDW = eval.RotateNew(ctBoot, classes*(cells*filters + filters + features*filters) + classes*filters)

	eval.MulRelin(ctDW, ctU1, ctDW)
	if err := eval.Rescale(ctDW, params.Scale(), ctDW); err != nil{
		panic(err)
	}

	ctDC = eval.RotateNew(ctDW, classes*filters)

	eval.InnerSum(ctDC, cells * filters + filters + features*filters, classes, ctDC)
	
	//fmt.Println("E1xW")
	//DecryptPrint(0, classes * (cells * filters + filters + features*filters), ctDC, params, sk)

	return ctDW, ctDC
}

func EncodeLabelsForBackward(Y *ckks.Matrix, cells, features, filters, classes int, params *ckks.Parameters) (*ckks.Plaintext){

	encoder := ckks.NewEncoder(params)

	values := make([]complex128, params.Slots())

	for i := 0; i < classes; i++ {
		c := Y.M[i]
		for j := 0; j < filters; j++ {
			values[i*filters + j] = c
		}
	}

	idx := classes * filters

	for i := 0; i < classes; i++ {
		c := Y.M[i]
		for j := 0; j <  (cells * filters + filters + features*filters); j++ {
			values[idx + i*(cells * filters + filters + features*filters) + j] = c
		}
	}

	pt := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	encoder.EncodeNTT(pt, values, params.LogSlots())

	return pt
}
