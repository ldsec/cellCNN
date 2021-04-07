package cellCNN

import(
	"github.com/ldsec/lattigo/v2/ckks"
)

func DeltaW(ctBoot *ckks.Ciphertext, Y *ckks.Plaintext, features, filters, classes int, params *ckks.Parameters, eval ckks.Evaluator, sk *ckks.SecretKey) (E1, DW *ckks.Ciphertext){

	// sigma(U) and sigma'(U)
	ctL, ctLDeriv := ActivationsCt(ctBoot, params.Scale(), eval)

	// Y - sigma(U)
	eval.Sub(Y, ctL, ctL)

	// E1 = sigma'(U) * (Y - sigma(U))
	eval.MulRelin(ctL, ctLDeriv, ctL)
	if err := eval.Rescale(ctL, params.Scale(), ctL); err != nil{
		panic(err)
	}

	// Ppool * E1 
	ctPpool := eval.RotateNew(ctBoot, classes*filters*features + classes*filters)


	eval.MulRelin(ctPpool, ctL, ctPpool)
	if err := eval.Rescale(ctPpool, params.Scale(), ctPpool); err != nil{
		panic(err)
	}

	E1 = ctL
	DW = ctPpool

	return E1, DW
}

func DeltaC(ctBoot, E1 *ckks.Ciphertext, ptLt []*ckks.Plaintext, cells, features, filters, classes int, params *ckks.Parameters, eval ckks.Evaluator, sk *ckks.SecretKey)(ctDC, ctE0 *ckks.Ciphertext){

	tmp := ckks.NewCiphertext(params, 1, E1.Level(), E1.Scale())
	
	// Puts [W0 ... W0][W1 ... W1] on the first slots
	eval.Rotate(ctBoot, classes*filters + classes*filters*features + classes * filters, ctBoot)
	eval.Rotate(E1, classes*filters, E1)
	
	// [W0 ... W0][W1 ... W1]
	//      X          X
	// [E0 ... E0][E1 ... E1]
	//      =          =
	// [E2 ... E2][E3 ... E3]
	eval.MulRelin(ctBoot, E1, ctBoot)
	eval.Rescale(ctBoot, params.Scale(), ctBoot)

	// [ E2 ...  E2][ E3 ...  E3]
	//       +            +
	// [iE2 ... iE2][iE3 ... iE3]
	//       =            =
	// [(E2+iE2) ... (E2+iE2)] [(E3+iE3) ... (E3+iE3)]
	eval.MultByi(ctBoot, tmp)
	eval.Add(ctBoot, tmp, ctBoot)

	eval.InnerSum(ctBoot, features*filters, classes, ctBoot)

	// MultSum with transpose(L)
	ctDC = eval.MulNew(ctBoot, ptLt[0])

	for i := 1; i < cells>>1; i++ {
		eval.Mul(ctBoot, ptLt[i], tmp) 
		eval.Add(ctDC, tmp, ctDC)
	}

	eval.Rescale(ctDC, params.Scale(), ctDC)

	// Eliminates the imaginary value
	eval.Add(ctDC, eval.ConjugateNew(ctDC), ctDC)

	// Replicates to match the encrypted convolution matrix
	//Replicate(ctDC, features*filters, int(float64(cells)/float64(features) + 1.5), eval)

	tmp = ctDC.CopyNew().Ciphertext()

	for i := 1; i < int(float64(cells)/float64(features) + 1.5); i++{
		eval.Rotate(tmp, -filters * features, tmp)
		eval.Add(ctDC, tmp, ctDC)
	}

	return ctDC, ctBoot
}

