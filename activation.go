package cellCNN

import(
	"github.com/ldsec/lattigo/v2/ckks"
)

//var coeffsActivation = []complex128{0.5, 0.1831, 0, -0.003817}

//Interval 3
var coeffsActivation = []complex128{0.5, 0.24656666666666668, 0, -0.009070370370370371}
var coeffsActivationDeriv = CoeffsDeriv(coeffsActivation)

func EvaluatePoly(x complex128, coeffs []complex128) (y complex128){
	y = coeffs[len(coeffs)-1]
	for i := len(coeffs)-1; i > 0; i--{
		y *= x
		y += coeffs[i-1]
	}
	return
}

func CoeffsDeriv(coeffs []complex128) (coeffsDeriv []complex128){

	coeffsDeriv = make([]complex128, len(coeffs)-1)

	for i := 1; i < len(coeffs); i++{
		coeffsDeriv[i-1] = coeffs[i] * complex(float64(i), 0)
	}

	return
}

func Activation(x complex128) complex128{
	return EvaluatePoly(x, coeffsActivation)
}

func ActivationDeriv(x complex128) complex128{
	return EvaluatePoly(x, coeffsActivationDeriv)
}

func ActivationsCt(ctU0 *ckks.Ciphertext, scale float64, eval ckks.Evaluator) (ctL1, ctL1Deriv *ckks.Ciphertext){
	var err error

	if ctL1, err = eval.EvaluatePoly(ctU0, ckks.NewPoly(coeffsActivation), scale); err != nil {
		panic(err)
	}

	if ctL1Deriv, err = eval.EvaluatePoly(ctU0, ckks.NewPoly(coeffsActivationDeriv), scale); err != nil {
		panic(err)
	}

	return
}