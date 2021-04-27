package cellCNN

import(
	"github.com/ldsec/lattigo/v2/ckks"
	"math"
)

func Backward(ctBoot *ckks.Ciphertext, Y, ptLBackward, maskPtW, maskPtC *ckks.Plaintext, cells, features, filters, classes int, params *ckks.Parameters, eval ckks.Evaluator, sk *ckks.SecretKey) (ctDW, ctDC *ckks.Ciphertext){


	// Extracts previous DC * momentum and previous DW * momentum
	ctDWPrev := eval.RotateNew(ctBoot, 2*(classes*filters + classes * filters * (cells + features + 1)))
	ctDCPrev := eval.RotateNew(ctBoot, 2*(classes*filters + classes * filters * (cells + features + 1)) + classes*filters)

	// sigma(U) and sigma'(U)
	ctU1, ctU1Deriv := ActivationsCt(ctBoot, params, eval)

	// Y - sigma(U)
	eval.Sub(Y, ctU1, ctU1)

	// E1 = sigma'(U) * (Y - sigma(U))
	eval.MulRelin(ctU1, ctU1Deriv, ctU1)
	if err := eval.Rescale(ctU1, params.Scale(), ctU1); err != nil{
		panic(err)
	}

	// Acess the index of the pooling results and upsampled W
	tmp := eval.RotateNew(ctBoot, classes*(cells*filters + filters + features*filters) + classes*filters)

	// Multiplies at the same time pool^t x E1 and upSampled(E1 x W^t)
	eval.MulRelin(tmp, ctU1, tmp)
	if err := eval.Rescale(tmp, params.Scale(), tmp); err != nil{
		panic(err)
	}

	// Adds prev DW*momentum
	ctDW = eval.AddNew(tmp, ctDWPrev)
	eval.Mul(ctDW, maskPtW, ctDW)
	eval.Rescale(ctDW, params.Scale(), ctDW)

	// Accesses upSampled(E1 x W^t)
	ctDC = eval.RotateNew(tmp, classes*filters)

	// Finishes the computation of E0 = upSampled(E1 x W^t) by summing all the rows
	eval.InnerSum(ctDC, cells * filters + filters + features*filters, classes, ctDC)

	//  DC = Ltranspose x E0
	eval.Mul(ctDC, ptLBackward, ctDC)
	eval.Rescale(ctDC, params.Scale(), ctDC)

	// adds prevDC * momentum
	eval.Mul(ctDCPrev, maskPtC, ctDCPrev)
	eval.Rescale(ctDCPrev, params.Scale(), ctDCPrev)
	eval.Add(ctDC, ctDCPrev, ctDC)

	return ctDW, ctDC
}

func EncodeCellsForBackward(L *ckks.Matrix, cells, features, filters, classes int, learningRate float64, params *ckks.Parameters) (*ckks.Plaintext){
	encoder := ckks.NewEncoder(params)

	values := make([]complex128, params.Slots())

	LSum := new(ckks.Matrix)
	LSum.SumRows(L.Transpose())


	idxLSum := 0

	for j := 0; j < cells*filters + filters + features*filters; j++ {

		if j%filters == 0 && j != 0{
			idxLSum++
			idxLSum%=features
		}

		c := real(LSum.M[idxLSum]) * math.Pow(learningRate / float64(cells), 0.5)

		values[j] = complex(c, 0)
	}

	pt := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	encoder.EncodeNTT(pt, values, params.LogSlots())

	return pt
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


func BackwardWithPrePooling(ctBoot *ckks.Ciphertext, Y, ptLBackward *ckks.Plaintext, features, filters, classes int, lastSample bool, params *ckks.Parameters, eval ckks.Evaluator, sk *ckks.SecretKey) (ctDC, ctDW, ctDCPrev, ctDWPrev  *ckks.Ciphertext){

	convolutionMatrixSize := ConvolutionMatrixSize(1, features, filters)
	denseMatrixSize := DenseMatrixSize(filters, classes)

	// ctBoot
	//[[        U        ][           U           ] [      Ppool       ] [     W transpose row encoded     ] [ Previous DeltaW ] [        Previous DeltaC        ] [ available ]]
	// | DenseMatrixSize || classes*ConvolutionMatrixSize | | DenseMatrixSize  | | classes * ConvolutionMatrixSize | [ DenseMatrixSize ] [classes * ConvolutionMatrixSize] |           |

	// Extracts previous DC * momentum and previous DW * momentum
	if lastSample {

		//[[       Previous DeltaC            ] [ available ] [        U        ][                U                ] [      Ppool       ] [     W transpose row encoded    ] [ Previous DeltaW ] ]
		// [ classes * ConvolutionMatrixSize  ] |           | | DenseMatrixSize || classes * ConvolutionMatrixSize | | DenseMatrixSize  | | classes * ConvolutionMatrixSize| [ DenseMatrixSize ] 
		ctDCPrev = eval.RotateNew(ctBoot, 3*denseMatrixSize + 2*classes * convolutionMatrixSize)

		//[[ Previous DeltaW ] [     Previous DeltaC           ] [ available ] [        U        ][                U                ] [      Ppool       ] [     W transpose row encoded    ] ]
		// [ DenseMatrixSize ] [classes * ConvolutionMatrixSize] |           | | DenseMatrixSize || classes * ConvolutionMatrixSize | | DenseMatrixSize  | | classes * ConvolutionMatrixSize| 
		ctDWPrev = eval.RotateNew(ctBoot, 2*denseMatrixSize + 2*classes * convolutionMatrixSize)
	}

	// sigma(U) and sigma'(U)
	ctU1, ctU1Deriv := ActivationsCt(ctBoot, params, eval)

	// Y - sigma(U)
	eval.Sub(Y, ctU1, ctU1)

	// E1 = sigma'(U) * (Y - sigma(U))
	eval.MulRelin(ctU1, ctU1Deriv, ctU1)
	if err := eval.Rescale(ctU1, params.Scale(), ctU1); err != nil{
		panic(err)
	}

	// Access the index of the pooling results and upsampled W

	//[[      Ppool       ] [     W transpose row encoded    ] [ Previous DeltaW ] [     Previous DeltaC           ] [ available ] [        U        ][                U                ] ]
	// | DenseMatrixSize  | | classes * ConvolutionMatrixSize| [ DenseMatrixSize ] [classes * ConvolutionMatrixSize] |           | | DenseMatrixSize || classes * ConvolutionMatrixSize | 
	ctDW = eval.RotateNew(ctBoot, denseMatrixSize + classes*convolutionMatrixSize)

	// Multiplies at the same time pool^t x E1 and upSampled(E1 x W^t)
	eval.MulRelin(ctDW, ctU1, ctDW)
	if err := eval.Rescale(ctDW, params.Scale(), ctDW); err != nil{
		panic(err)
	}

	// Accesses upSampled(E1 x W^t)
	ctDC = eval.RotateNew(ctDW, denseMatrixSize)

	// Finishes the computation of E0 = upSampled(E1 x W^t) by summing all the rows
	eval.InnerSum(ctDC, convolutionMatrixSize, classes, ctDC)

	//  DC = Ltranspose x E0
	eval.Mul(ctDC, ptLBackward, ctDC)
	eval.Rescale(ctDC, params.Scale(), ctDC)

	return ctDC, ctDW, ctDCPrev, ctDWPrev 
}

func EncodeCellsForBackwardWithPrepool(level int, L *ckks.Matrix, features, filters, classes int, learningRate float64, params *ckks.Parameters) (*ckks.Plaintext){

	convolutionMatrixSize := ConvolutionMatrixSize(1, features, filters)

	encoder := ckks.NewEncoder(params)

	values := make([]complex128, params.Slots())

	idxLSum := 0

	for j := 0; j < convolutionMatrixSize; j++ {

		if j%filters == 0 && j != 0{
			idxLSum++
			idxLSum%=features
		}

		c := real(L.M[idxLSum]) * learningRate

		values[j] = complex(c, 0)
	}

	pt := ckks.NewPlaintext(params, level, float64(params.Qi()[level]))
	encoder.EncodeNTT(pt, values, params.LogSlots())

	return pt
}

func EncodeLabelsForBackwardWithPrepooling(Y *ckks.Matrix, features, filters, classes int, params *ckks.Parameters) (*ckks.Plaintext){

	convolutionMatrixSize := ConvolutionMatrixSize(1, features, filters)

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
		for j := 0; j <  convolutionMatrixSize; j++ {
			values[idx + i*convolutionMatrixSize + j] = c
		}
	}

	pt := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	encoder.EncodeNTT(pt, values, params.LogSlots())

	return pt
}
