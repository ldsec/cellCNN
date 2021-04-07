package cellCNN

import(
	"github.com/ldsec/lattigo/v2/ckks"
)

func Backward(ctBoot *ckks.Ciphertext, Y *ckks.Plaintext, ptLt []*ckks.Plaintext, cells, features, filters, classes int, params *ckks.Parameters, eval ckks.Evaluator, sk *ckks.SecretKey) (ctDW, ctDC *ckks.Ciphertext){


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
	ctDW = eval.RotateNew(ctBoot, classes*filters*features + classes*filters)

	eval.MulRelin(ctDW, ctU1, ctDW)
	if err := eval.Rescale(ctDW, params.Scale(), ctDW); err != nil{
		panic(err)
	}

	E1 := eval.RotateNew(ctDW, classes*filters)

	tmp := ckks.NewCiphertext(params, 1, E1.Level(), E1.Scale())

	eval.MultByi(E1, tmp)

	eval.Add(E1, tmp, E1)

	eval.InnerSum(E1, features*filters, classes, E1)

	// MultSum with transpose(L)
	ctDC = eval.MulNew(E1, ptLt[0])

	for i := 1; i < cells>>1; i++ {
		eval.Mul(E1, ptLt[i], tmp) 
		eval.Add(ctDC, tmp, ctDC)
	}

	eval.Rescale(ctDC, params.Scale(), ctDC)

	// Eliminates the imaginary value
	eval.Add(ctDC, eval.ConjugateNew(ctDC), ctDC)

	// Replicates to match the encrypted convolution matrix
	// Replicate(ctDC, features*filters, int(float64(cells)/float64(features) + 1.5), eval)
	tmp = ctDC.CopyNew().Ciphertext()

	for i := 1; i < int(float64(cells)/float64(features) + 1.5); i++{
		eval.Rotate(tmp, -filters * features, tmp)
		eval.Add(ctDC, tmp, ctDC)
	}

	return 
}

func EncodeLabelsForBackward(Y *ckks.Matrix, features, filters, classes int, params *ckks.Parameters) (*ckks.Plaintext){

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
		for j := 0; j < filters*features; j++ {
			values[idx + i*filters*features + j] = c
		}
	}

	pt := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	encoder.EncodeNTT(pt, values, params.LogSlots())

	return pt
}
