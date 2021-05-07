package cellCNN

import(
	"github.com/ldsec/lattigo/v2/ckks"
	"math"
	"fmt"
)

func Backward(ctBoot *ckks.Ciphertext, Y, ptLBackward, maskPtW, maskPtC *ckks.Plaintext, cells, features, filters, classes int, params *ckks.Parameters, eval ckks.Evaluator, sk *ckks.SecretKey) (ctDW, ctDC *ckks.Ciphertext){


	// Extracts previous DC * momentum and previous DW * momentum
	ctDWPrev := eval.RotateNew(ctBoot, 2*(classes*filters + classes * filters * (cells + features + 1)))
	ctDCPrev := eval.RotateNew(ctBoot, 2*(classes*filters + classes * filters * (cells + features + 1)) + classes*filters)

	// sigma(U) and sigma'(U)
	ctU1, ctU1Deriv := ActivationsCt(ctBoot, params, eval)

	// Y - sigma(U)
	eval.Sub(ctU1, Y, ctU1)

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




func EncodeLabelsForBackwardWithPrepooling(Y *ckks.Matrix, features, filters, classes int, params *ckks.Parameters) (*ckks.Plaintext){

	convolutionMatrixSize := ConvolutionMatrixSize(Y.Rows(), features, filters)

	encoder := ckks.NewEncoder(params)

	values := make([]complex128, params.Slots())

	
	for i := 0; i < len(Y.M); i++ {
		c := Y.M[i]
		for j := 0; j < filters; j++ {
			values[i*filters + j] = c
		}
	}
	

	idx := Y.Rows() * classes * filters

	if false {
		fmt.Println("Y Plaintext")
		fmt.Printf("[\n")
		for i := 0; i < len(Y.M); i++ {
			fmt.Printf("[ ")
			for j := 0; j < filters; j++ {
				fmt.Printf("%11.8f, ", real(values[i*filters+j]))

			}
			fmt.Printf("],\n")
		}
		fmt.Printf("]\n")
		fmt.Println()
	}

	for i := 0; i < classes; i++ {
		pos := 0
		for j := 0; j <  convolutionMatrixSize; j++ {
			c := Y.M[((pos*classes)%len(Y.M))+i]

			if (j+1)%filters == 0{
				pos++
			}
			values[idx + i*convolutionMatrixSize + j] = c
		}
	}

	idx += classes * convolutionMatrixSize

	pt := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	encoder.EncodeNTT(pt, values, params.LogSlots())

	return pt
}

func EncodeCellsForBackwardWithPrepooling(level int, L *ckks.Matrix, batchSize, features, filters, classes int, learningRate float64, params *ckks.Parameters) (*ckks.Plaintext){

	convolutionMatrixSize := ConvolutionMatrixSize(batchSize, features, filters)

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


