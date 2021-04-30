package main

import (
	"fmt"
	"time"
	"math"
	"math/rand"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/cellCNN/cellCNN_JPB"
)




func main() {

	trainEncrypted := true

	params := cellCNN.GenParams()

	fmt.Println(params.LogQP())

	// Keys
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()

	// Relinearization key
	rlk := kgen.GenRelinearizationKey(sk)

	encoder := ckks.NewEncoder(params)

	batchSize := cellCNN.BatcheSize
	samples := cellCNN.Samples
	cells := cellCNN.Cells
	features := cellCNN.Features
	filters := cellCNN.Filters
	classes := cellCNN.Classes


	denseMatrixSize := cellCNN.DenseMatrixSize(filters, classes)
	convolutionMatrixSize := cellCNN.ConvolutionMatrixSize(batchSize, features, filters)


	slotUsage := 3*batchSize*denseMatrixSize + (2*classes+1) * convolutionMatrixSize

	fmt.Printf("Slots Usage : %d/%d \n", slotUsage, params.Slots()) 

	learningRate := cellCNN.LearningRate
	momentum := cellCNN.Momentum

	rotations := []int{}

	levelW := 2 + 1
	levelC := 2 + 2

	if trainEncrypted {

		rotations = append(rotations, filters)

		// Convolution rotations
		for i := 1; i < features>>1; i++ {
			rotations = append(rotations, 2*filters*i)
		}

		for i := 1; i < batchSize>>1; i++ {
			rotations = append(rotations, 2*filters*i)
		}

		// Dense layer rotations
		rotations = append(rotations, kgen.GenRotationIndexesForInnerSum(1, filters)...)

		rotations = append(rotations, kgen.GenRotationIndexesForInnerSum(convolutionMatrixSize, classes)...)

		rotations = append(rotations, kgen.GenRotationIndexesForInnerSum(batchSize, classes)...)

		rotations = append(rotations, kgen.GenRotationIndexesForInnerSum(classes*filters, batchSize)...)

		// Pre-pool convolution replication
		rotations = append(rotations, -batchSize*filters)

		// Repacking of ctPpool before bootstrapping

		rotations = append(rotations,  1*batchSize*denseMatrixSize)
		rotations = append(rotations, -1*batchSize*denseMatrixSize)
		rotations = append(rotations, -2*batchSize*denseMatrixSize)
		rotations = append(rotations, -3*batchSize*denseMatrixSize)
		rotations = append(rotations, -4*batchSize*denseMatrixSize)

		rotations = append(rotations, 1*batchSize*denseMatrixSize + 1*classes*convolutionMatrixSize)
		rotations = append(rotations, 1*batchSize*denseMatrixSize + 2*classes*convolutionMatrixSize)
		rotations = append(rotations, 2*batchSize*denseMatrixSize + 2*classes*convolutionMatrixSize)
		rotations = append(rotations, 3*batchSize*denseMatrixSize + 2*classes*convolutionMatrixSize)

		rotations = append(rotations, -batchSize*filters + filters)

		// Replication of DC
		rotations = append(rotations, kgen.GenRotationIndexesForReplicate(features*filters, int(math.Ceil(float64(convolutionMatrixSize)/float64(features * filters))))...)

		// Replication of DW
		rotations = append(rotations, kgen.GenRotationIndexesForReplicate(filters, batchSize)...)
	}
	
	rotkeys := kgen.GenRotationKeysForRotations(rotations, true, sk)

	eval := ckks.NewEvaluator(params, ckks.EvaluationKey{rlk, rotkeys})

	fmt.Println("Loading Data ...")
	XTrain, YTrain := cellCNN.LoadTrainDataFrom("../normalized/", 2000, cellCNN.Cells, cellCNN.Features)

	WeightsC := []float64{-0.157606, -0.042193, 0.090079, -0.140268, 0.127641, -0.163260, 
		0.123652, 0.045518, -0.080589, -0.145210, -0.016246, 0.038132, 
		0.017790, -0.064759, 0.161123, -0.048306, 0.139825, -0.149952, 
		-0.084128, 0.028893, 0.023134, -0.009103, 0.060570, -0.034424, 
		-0.157708, -0.001613, -0.019875, -0.115267, -0.005837, -0.016422, 
		0.009624, -0.005860, -0.163596, 0.054042, -0.109609, 0.154910, 
		-0.105088, -0.113516, -0.001531, -0.013902, -0.155277, -0.026212, 
		0.103340, 0.051511, -0.045778, 0.060698, -0.162107, -0.016309, 
		0.016753, 0.008481, 0.117370, -0.009266, -0.074229, -0.086144, 
		-0.007862, 0.033828, -0.160092, 0.034801, 0.128095, 0.127478, 
		0.012857, 0.086050, -0.101835, -0.153228, 0.034571, -0.082220, 
		-0.010461, 0.132321, 0.092869, 0.019801, 0.132878, -0.053578, 
		-0.099425, 0.003100, 0.123340, 0.015970, 0.067966, 0.086562, 
		0.099241, -0.131986, -0.123341, 0.114430, 0.056390, -0.082455, 
		0.154547, 0.139452, -0.014389, 0.130446, 0.116000, -0.083244, 
		-0.072166, -0.031328, -0.099770, 0.095270, 0.004664, 0.142161, 
		0.088048, 0.050235, -0.004322, 0.049053, 0.023691, 0.077500, 
		0.069644, -0.043004, -0.130544, 0.088716, 0.070841, -0.117993, 
		-0.043149, -0.072074, 0.138816, -0.011218, -0.067313, 0.043315, 
		-0.039192, -0.091592, 0.080425, -0.013404, 0.161842, -0.048344, 
		-0.076622, -0.128067, -0.065961, -0.152914, -0.099298, -0.030963, 
		-0.083517, 0.114592, 0.017944, 0.078460, 0.039129, -0.083513, 
		0.154834, -0.081162, -0.126272, -0.135852, -0.109560, 0.123894, 
		0.035337, 0.056845, 0.005921, 0.141798, -0.149847, 0.161548, 
		0.071480, 0.078476, -0.122914, 0.157709, 0.036181, 0.141595, 
		0.070096, 0.064475, 0.128670, 0.155279, 0.011736, -0.164367, 
		-0.161145, -0.099134, 0.141477, -0.140490, -0.123246, -0.090162, 
		0.053416, -0.046566, 0.014807, -0.153300, -0.142962, 0.135505, 
		0.143485, -0.078104, -0.034584, 0.066045, -0.105522, 0.149281, 
		0.068742, -0.053235, -0.128386, -0.037187, 0.088665, -0.105581, 
		-0.054910, -0.088041, -0.097207, 0.047082, -0.046825, -0.049848, 
		0.085329, 0.013615, -0.091482, 0.056405, -0.139830, 0.049000, 
		0.002704, -0.150111, -0.150939, 0.126627, 0.035670, 0.070295, 
		-0.076432, 0.023660, -0.163961, -0.040700, 0.102170, 0.064880, 
		-0.075007, -0.103649, 0.144220, 0.009432, 0.108479, 0.018681, 
		0.099697, -0.038130, -0.009199, -0.121160, 0.136172, -0.118878, 
		0.045402, 0.114020, 0.059185, 0.041708, 0.046495, 0.117250, 
		0,0,0,0,0,0}
	WeightsD :=[]float64{0.270457, -0.004343, 0.290110, 0.172072, -0.184941, -0.075417, 
	0.334569, 0.362849, -0.001117, -0.172576, 0.388567, -0.038715}

	C := cellCNN.WeightsInit(features, filters, features)
	W := cellCNN.WeightsInit(filters, classes, filters) 

	for i := range C.M{
		C.M[i] = complex(WeightsC[i], 0)
	}

	for i := range W.M{
		W.M[i] = complex(WeightsD[i], 0)
	}


	fmt.Println("Done")

	ctC := cellCNN.EncryptRightForPtMul(C, batchSize, 1, params, levelC, sk)

	// Returns
	//
	// [[ W transpose row encoded ] [         available         ]]
	//  |    classes * filters    | | Slots - classes * filters | 
	//
	ctW := cellCNN.EncryptRightForNaiveMul(W, batchSize, params, levelW, sk)


	levelMaskPtW := 5
	levelMaskPtC := 4

	scaleMaskPtW := float64(params.Qi()[levelMaskPtW])
	scaleMaskPtC := float64(params.Qi()[levelMaskPtC])

	// Mask W
	maskW := make([]complex128, params.Slots())

	// mask
	for i := 0; i < batchSize*denseMatrixSize; i++ {
		maskW[i] = complex(1.0, 0)
	}
	maskPtW := ckks.NewPlaintext(params, levelMaskPtW, scaleMaskPtW)
	encoder.EncodeNTT(maskPtW, maskW, params.LogSlots())

	// mask w0
	maskW = make([]complex128, params.Slots())
	for i := 0; i < denseMatrixSize>>1; i++ {
		maskW[i] = complex(1.0, 0)
		maskW[i+(denseMatrixSize>>1)] = complex(0, 0)
	}
	maskPtW0 := ckks.NewPlaintext(params, levelMaskPtW, scaleMaskPtW)
	encoder.EncodeNTT(maskPtW0, maskW, params.LogSlots())

	// mask w1
	for i := 0; i < denseMatrixSize>>1; i++ {
		maskW[i] = complex(0, 0)
		maskW[(denseMatrixSize>>1)+i] = complex(1.0, 0)
	}

	maskPtW1 := ckks.NewPlaintext(params, levelMaskPtW, scaleMaskPtW)
	encoder.EncodeNTT(maskPtW1, maskW, params.LogSlots())

	// Mask C
	maskC := make([]complex128, params.Slots())

	// mask half
	for i := 0; i < convolutionMatrixSize; i++ {
		maskC[i] = complex(1.0, 0)
	}
	maskPtC := ckks.NewPlaintext(params, levelMaskPtC, scaleMaskPtC)
	encoder.EncodeNTT(maskPtC, maskC, params.LogSlots())

	// mask
	for i := 0; i < convolutionMatrixSize; i++ {
		maskC[i] = complex(0.5, 0)
	}
	maskPtCHalf := ckks.NewPlaintext(params, levelMaskPtC, scaleMaskPtC)
	encoder.EncodeNTT(maskPtCHalf, maskC, params.LogSlots())

	var ctDW, ctDC *ckks.Ciphertext

	XPrePool := new(ckks.Matrix)

	PoolBatch := new(ckks.Matrix)
	UBatch := new(ckks.Matrix)
	L1Batch := new(ckks.Matrix)
	L1DerivBatch := new(ckks.Matrix)
	E0Batch := new(ckks.Matrix)
	E1Batch := new(ckks.Matrix)
	DCBatch := new(ckks.Matrix)
	DWBatch := new(ckks.Matrix)
	

	DCPrevBatch := ckks.NewMatrix(features, filters)
	DWPrevBatch := ckks.NewMatrix(filters, classes)
	ctDCPrev := ckks.NewCiphertext(params, 1, 3, params.Scale())
	ctDWPrev := ckks.NewCiphertext(params, 1, 4, params.Scale())
	var ctDCPrevBoot, ctDWPrevBoot *ckks.Ciphertext

	epoch := 15
	niter := epoch * samples / batchSize

	fmt.Prinf("#Iters : %d\n", niter)

	for i := 0; i < niter; i++{

		XBatch := ckks.NewMatrix(batchSize, features)
		YBatch := ckks.NewMatrix(batchSize, classes)

		// Pre-pools the cells
		for j := 0; j < batchSize; j++ {

			randi := rand.Intn(2000)

			X := XTrain[randi]
			Y := YTrain[randi]

			XPrePool.SumColumns(X)
			XPrePool.MultConst(XPrePool, complex(1.0/float64(cells), 0))

			XBatch.SetRow(j, XPrePool.M)
			YBatch.SetRow(j, []complex128{Y.M[1], Y.M[0]})
		}

		// === Plaintext ===

		// Convolution
		PoolBatch.MulMat(XBatch, C)

		// Dense
		UBatch.MulMat(PoolBatch, W)

		// Activations
		L1Batch.Func(UBatch, cellCNN.Activation)
		L1DerivBatch.Func(UBatch, cellCNN.ActivationDeriv)

		// Dense error
		E1Batch.Sub(L1Batch, YBatch)
		E1Batch.Dot(E1Batch, L1DerivBatch)

		// Convolution error
		E0Batch.MulMat(E1Batch, W.Transpose())

		// Updated weights
		DWBatch.MulMat(PoolBatch.Transpose(), E1Batch)
		DCBatch.MulMat(XBatch.Transpose(), E0Batch)

		// Takes the average
		DWBatch.MultConst(DWBatch, complex(learningRate, 0))
		DCBatch.MultConst(DCBatch, complex(learningRate, 0))

		// Adds the previous weights
		// W_i = learning_rate * Wt + W_i-1 * momentum
		DWBatch.Add(DWBatch, DWPrevBatch)
		DCBatch.Add(DCBatch, DCPrevBatch)

		// Stores the current weights
		// W_i = learning_rate * Wt + W_i-1 * momentum
		DWPrevBatch.MultConst(DWBatch, complex(momentum, 0))
		DCPrevBatch.MultConst(DCBatch, complex(momentum, 0))
		
		// Updates the matrices
		W.Sub(W, DWBatch)
		C.Sub(C, DCBatch)

		// === Ciphertext === 
		if trainEncrypted{

			ptL := cellCNN.EncodeLeftForPtMul(XBatch, filters, 1.0, params)
			ptY := cellCNN.EncodeLabelsForBackwardWithPrepooling(YBatch, features, filters, classes, params)
			ptLBackward := cellCNN.EncodeLeftForPtMul(XBatch.Transpose(), filters, learningRate, params) 

			start := time.Now()

			ctTmp := cellCNN.ForwardWithPrepooling(ptL, ctC, ctW, ctDCPrev, ctDWPrev, batchSize, features, filters, classes, eval, params, sk)
			
			ctBoot := cellCNN.DummyBootWithPrepooling(ctTmp, batchSize, features, filters, classes, learningRate, momentum, params, sk)

			ctDC, ctDW, ctDCPrevBoot, ctDWPrevBoot = cellCNN.BackwardWithPrePooling(ctBoot, ptY, ptLBackward, batchSize, features, filters, classes, params, eval, sk)

			// Cleans the imaginary part
			eval.Add(ctDC, eval.ConjugateNew(ctDC), ctDC)

			// Replicates ctDC so that it is at least as large as convolutionMatrixSize
			eval.Replicate(ctDC, features*filters, int(math.Ceil(float64(convolutionMatrixSize)/float64(features * filters))), ctDC)

			// Divides by the average and learning rate and cleans the non-desired slots
			eval.Mul(ctDC, maskPtCHalf, ctDC)

			// Divides by the average, masks the values and extract the first and second classe
			ctDWtmp := eval.MulNew(ctDW, maskPtW0)
			eval.Mul(ctDW, maskPtW1, ctDW)

			// Replicates DW batch times (no masking needed as it is a multiple of filters)
			eval.Rotate(ctDW, -batchSize*filters + filters, ctDW)
			eval.Add(ctDW, ctDWtmp, ctDW)
			eval.Replicate(ctDW, filters, batchSize, ctDW)

			// Mask DWPrev*momentum and DCPrev*momentum
			eval.Mul(ctDCPrevBoot, maskPtC, ctDCPrevBoot)
			eval.Mul(ctDWPrevBoot, maskPtW, ctDWPrevBoot)

			// Adds DW with DWPrev*momentum 
			eval.Add(ctDC, ctDCPrevBoot, ctDC)
			eval.Add(ctDW, ctDWPrevBoot, ctDW)

			// Rescales
			eval.Rescale(ctDC, params.Scale(), ctDC)
			eval.Rescale(ctDW, params.Scale(), ctDW)

			// Stores DW + DWPrev*momentum 
			ctDCPrev = ctDC.CopyNew().Ciphertext()
			ctDWPrev = ctDW.CopyNew().Ciphertext()

			// Updates the weights
			eval.Sub(ctC, ctDC, ctC)
			eval.Sub(ctW, ctDW, ctW)

			fmt.Printf("Iter[%02d] : %s\n", i, time.Since(start))
		}
	}

	fmt.Println("DC")
	DCBatch.Print()
	cellCNN.DecryptPrint(convolutionMatrixSize/filters+1, filters, true, ctDC, params, sk)

	fmt.Println("DW")
	DWBatch.Transpose().Print()
	cellCNN.DecryptPrint(batchSize*classes+1, filters, true, ctDW, params, sk)

	// Tests resuls :

	err := 0
	var v int
	nTests := 2000
	for i := 0; i < nTests; i++{
		X := XTrain[i]
		Y := YTrain[i]

		XPrePool.SumColumns(X)
		XPrePool.MultConst(XPrePool, complex(1.0/float64(cells), 0))

		// Convolution
		PoolBatch.MulMat(XPrePool, C)
		// Dense
		UBatch.MulMat(PoolBatch, W)
		// Activations
		L1Batch.Func(UBatch, cellCNN.Activation)

		if real(L1Batch.M[0]) > real(L1Batch.M[1]){
			v = 1
		}else{
			v = 0
		}

		if v != int(real(Y.M[1])){
			err++
		}
	}

	fmt.Println("error : ", float64(err)/float64(nTests))

}


