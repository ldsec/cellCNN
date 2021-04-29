package main

import (
	"fmt"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/cellCNN/cellCNN_JPB"
)




func main() {

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

	learningRate := cellCNN.LearningRate
	momentum := cellCNN.Momentum

	rotations := []int{}

	rotations = append(rotations, filters)

	levelW := 2 + 1
	levelC := 2 + 2

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

	rotations = append(rotations,  1*batchSize * denseMatrixSize)
	rotations = append(rotations, -1*batchSize * denseMatrixSize)
	rotations = append(rotations, -2*batchSize * denseMatrixSize)
	rotations = append(rotations, -3*batchSize * denseMatrixSize)
	rotations = append(rotations, -4*batchSize * denseMatrixSize)

	rotations = append(rotations, batchSize*classes*filters + classes * convolutionMatrixSize)

	rotations = append(rotations, 3*denseMatrixSize + 2*classes*convolutionMatrixSize)
	rotations = append(rotations, 2*denseMatrixSize + 2*classes*convolutionMatrixSize)

	

	rotkeys := kgen.GenRotationKeysForRotations(rotations, true, sk)

	eval := ckks.NewEvaluator(params, ckks.EvaluationKey{rlk, rotkeys})
	_=eval

	XTrain, YTrain := cellCNN.LoadTrainDataFrom("../normalized/", samples, cellCNN.Cells, cellCNN.Features)
	C := cellCNN.WeightsInit(features, filters, features)
	W := cellCNN.WeightsInit(filters, classes, filters) 

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
	for i := 0; i < classes*filters; i++ {
		maskW[i] = complex(1.0, 0)
	}
	maskPtW := ckks.NewPlaintext(params, levelMaskPtW, scaleMaskPtW)
	encoder.EncodeNTT(maskPtW, maskW, params.LogSlots())

	// mask avg
	for i := 0; i < classes*filters; i++ {
		maskW[i] = complex(1.0/float64(batchSize), 0)
	}
	maskPtWavg := ckks.NewPlaintext(params, levelMaskPtW, scaleMaskPtW)
	encoder.EncodeNTT(maskPtWavg, maskW, params.LogSlots())

	// Mask C
	maskC := make([]complex128, params.Slots())

	// mask
	for i := 0; i < cells * filters + (features/2 -1)*2*filters + filters; i++ {
		maskC[i] = complex(1.0, 0)
	}
	maskPtC := ckks.NewPlaintext(params, levelMaskPtC, scaleMaskPtC)
	encoder.EncodeNTT(maskPtC, maskC, params.LogSlots())

	// mask with avg
	for i := 0; i < cells * filters + (features/2 -1)*2*filters + filters; i++ {
		maskC[i] = complex(1.0/float64(batchSize), 0)
	}
	maskPtCavg := ckks.NewPlaintext(params, levelMaskPtC, scaleMaskPtC)
	encoder.EncodeNTT(maskPtCavg, maskC, params.LogSlots())

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

	epoch := 1
	for i := 0; i < epoch; i++{

		fmt.Printf("Epoch[%02d]\n", i)

		for k := 0; k < samples/batchSize; k++ {

			XBatch := ckks.NewMatrix(batchSize, features)
			YBatch := ckks.NewMatrix(batchSize, classes)

			// Pre-pools the cells
			for j := 0; j < batchSize; j++ {

				X := XTrain[k*batchSize + j]
				Y := YTrain[k*batchSize + j]

				XPrePool.SumColumns(X)
				XPrePool.MultConst(XPrePool, complex(1.0/float64(cells), 0))

				XBatch.SetRow(j, XPrePool.M)
				YBatch.SetRow(j, Y.M)
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
			E1Batch.Sub(YBatch, L1Batch)
			E1Batch.Dot(E1Batch, L1DerivBatch)

			// Convolution error
			E0Batch.MulMat(E1Batch, W.Transpose())

			// Updated weights
			DWBatch.MulMat(PoolBatch.Transpose(), E1Batch)
			DCBatch.MulMat(XBatch.Transpose(), E0Batch)

			fmt.Println("DW")
			DWBatch.Transpose().Print()

			fmt.Println("DC")
			DCBatch.Print()

			// Takes the average
			DWBatch.MultConst(DWBatch, complex(learningRate/float64(batchSize), 0))
			DCBatch.MultConst(DCBatch, complex(learningRate/float64(batchSize), 0))

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

			ptL := cellCNN.EncodeLeftForPtMul(XBatch, filters, 1.0, params)
			ptY := cellCNN.EncodeLabelsForBackwardWithPrepooling(YBatch, features, filters, classes, params)
			ptLBackward := cellCNN.EncodeLeftForPtMul(XBatch.Transpose(), filters, learningRate, params) //cellCNN.EncodeCellsForBackwardWithPrepooling(levelMaskPtW, XBatch.Transpose(), batchSize, features, filters, classes, learningRate, params)

			start := time.Now()

			ctTmp := cellCNN.ForwardWithPrepooling(ptL, ctC, ctW, ctDCPrev, ctDWPrev, batchSize, features, filters, classes, eval, params, sk)
			
			ctBoot := cellCNN.DummyBootWithPrepooling(ctTmp, batchSize, features, filters, classes, learningRate, momentum, params, sk)

			ctDC, ctDW, ctDCPrevBoot, ctDWPrevBoot = cellCNN.BackwardWithPrePooling(ctBoot, ptY, ptLBackward, batchSize, features, filters, classes, params, eval, sk)

			// Masks DW and DC
			ctDCAvg := eval.MulNew(ctDC, maskPtCavg)
			ctDWAvg := eval.MulNew(ctDW, maskPtWavg)

			// Mask DWPrev*momentum and DCPrev*momentum
			eval.Mul(ctDCPrevBoot, maskPtC, ctDCPrevBoot)
			eval.Mul(ctDWPrevBoot, maskPtW, ctDWPrevBoot)

			// Adds DW with DWPrev*momentum 
			//eval.Add(ctDCAvg, ctDCPrevBoot, ctDCAvg)
			//eval.Add(ctDWAvg, ctDWPrevBoot, ctDWAvg)

			// Rescales
			eval.Rescale(ctDCAvg, params.Scale(), ctDCAvg)
			eval.Rescale(ctDWAvg, params.Scale(), ctDWAvg)

			// Stores DW + DWPrev*momentum 
			ctDCPrev = ctDCAvg.CopyNew().Ciphertext()
			ctDWPrev = ctDWAvg.CopyNew().Ciphertext()

			// Updates the weights
			eval.Sub(ctC, ctDCAvg, ctC)
			eval.Sub(ctW, ctDWAvg, ctW)

			fmt.Printf("Batch[%02d-|%02d-%02d] : %s\n", k, k*batchSize, (k+1)*batchSize, time.Since(start))

			//cellCNN.DecryptPrintMatrix(C, true, ctC, params, sk)
			//cellCNN.DecryptPrintMatrix(W.Transpose(), true, ctW, params, sk)
		}
	}
}


