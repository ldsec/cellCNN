package main

import (
	"fmt"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/cellCNN"
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
	convolutionMatrixSize := cellCNN.ConvolutionMatrixSize(1, features, filters)

	learningRate := cellCNN.LearningRate
	momentum := cellCNN.Momentum

	rotations := []int{}

	rotations = append(rotations, filters)

	levelW := 2 + 1
	levelC := 2 + 2

	// Convolution rotations
	rotHoisted := []int{}
	for i := 1; i < features>>1; i++ {
		rotHoisted = append(rotHoisted, 2*filters*i)
	}

	rotations = append(rotations, rotHoisted...)

	// Pooling rotations
	rotations = append(rotations, kgen.GenRotationIndexesForInnerSum(filters, cells)...)
	for i := 1; i < classes; i++ {
		rotations = append(rotations, -filters*i)
	}

	// Dense layer rotations
	rotations = append(rotations, kgen.GenRotationIndexesForInnerSum(1, filters)...)

	
	//
	rotations = append(rotations, kgen.GenRotationIndexesForInnerSum(cells*filters + filters + features*filters, classes)...)

	rotations = append(rotations, kgen.GenRotationIndexesForInnerSum(convolutionMatrixSize, classes)...)

	// Pre-pool convolution replication
	rotations = append(rotations, -filters)

	// Repacking of ctPpool before bootstrapping

	// Repacking of ctW before bootstrapping
	rotations = append(rotations, classes*filters*features + classes*filters)

	rotations = append(rotations, -classes*filters)
	rotations = append(rotations, -2*classes*filters)
	rotations = append(rotations, -3*classes*filters)
	rotations = append(rotations, -4*classes*filters)

	rotations = append(rotations, features*filters)
	rotations = append(rotations, -features*filters)

	rotations = append(rotations, classes*filters + classes*filters*features + classes * filters)

	rotations = append(rotations, classes*(cells*filters + filters + features*filters) + classes*filters)

	rotations = append(rotations, denseMatrixSize + classes*convolutionMatrixSize)

	rotations = append(rotations, 3*denseMatrixSize + 2*classes*convolutionMatrixSize)
	rotations = append(rotations, 2*denseMatrixSize + 2*classes*convolutionMatrixSize)

	rotkeys := kgen.GenRotationKeysForRotations(rotations, true, sk)

	eval := ckks.NewEvaluator(params, ckks.EvaluationKey{rlk, rotkeys})
	_=eval

	XTrain, YTrain := cellCNN.LoadTrainDataFrom("../normalized/", samples, cellCNN.Cells, cellCNN.Features)
	C := cellCNN.WeightsInit(features, filters, features)
	W := cellCNN.WeightsInit(filters, classes, filters) 

	ctC := cellCNN.EncryptRightForPtMul(C, 1, params, levelC, sk)

	// Returns
	//
	// [[ W transpose row encoded ] [         available         ]]
	//  |    classes * filters    | | Slots - classes * filters | 
	//
	ctW := cellCNN.EncryptRightForNaiveMul(W, params, levelW, sk)


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
	Ppool := new(ckks.Matrix)
	U := new(ckks.Matrix)
	L1 := new(ckks.Matrix)
	L1Deriv := new(ckks.Matrix)
	E1 := new(ckks.Matrix)
	E0 := new(ckks.Matrix)
	DC := new(ckks.Matrix)
	DW := new(ckks.Matrix)

	DCPrev := ckks.NewMatrix(features, filters)
	DWPrev := ckks.NewMatrix(filters, classes)
	ctDCPrev := ckks.NewCiphertext(params, 1, 3, params.Scale())
	ctDWPrev := ckks.NewCiphertext(params, 1, 4, params.Scale())
	var ctDCPrevBoot, ctDWPrevBoot *ckks.Ciphertext

	epoch := 10
	for i := 0; i < epoch; i++{

		for k := 0; k < samples/batchSize; k++ {

			DCPool := ckks.NewMatrix(features, filters)
			DWPool := ckks.NewMatrix(filters, classes)

			ctDCPool := ckks.NewCiphertext(params, 1, params.MaxLevel(), params.Scale())
			ctDWPool := ckks.NewCiphertext(params, 1, params.MaxLevel(), params.Scale())

			fmt.Printf("Epoch[%02d] | Batch[%d]\n", i, k)

			for j := 0; j < batchSize; j++ {

				X := XTrain[k*batchSize + j]
				Y := YTrain[k*batchSize + j]

				XPrePool.SumColumns(X)
				XPrePool.MultConst(XPrePool, complex(1.0/float64(cells), 0))

				ptL := cellCNN.EncodeLeftForPtMul(XPrePool, filters, 1, params)
				ptY := cellCNN.EncodeLabelsForBackwardWithPrepooling(Y,features, filters, classes, params)
				ptLBackward := cellCNN.EncodeCellsForBackwardWithPrepool(levelMaskPtW, XPrePool.Transpose(), features, filters, classes, learningRate, params)

				// =======================================
				// ========== Plaintext circuit ==========
				// =======================================

				Ppool.MulMat(XPrePool, C)

				U.MulMat(Ppool, W)

				L1.Func(U, cellCNN.Activation)
				L1Deriv.Func(U, cellCNN.ActivationDeriv)

				E1.Sub(Y, L1)
				E1.Dot(E1, L1Deriv)

				DW.MulMat(Ppool.Transpose(), E1)

				E0.MulMat(E1, W.Transpose())

				DC.MulMat(XPrePool.Transpose(), E0)

				// Pools the gradients of the batch
				DCPool.Add(DCPool, DC)
				DWPool.Add(DWPool, DW)
				
				// =======================================

				start := time.Now()

				var ctTmp *ckks.Ciphertext
				if j == batchSize-1{
					ctTmp = cellCNN.ForwardWithPrepool(ptL, ctC, ctW, ctDCPrev, ctDWPrev, cells, features, filters, classes, eval, params, sk)
				}else{
					ctTmp = cellCNN.ForwardWithPrepool(ptL, ctC, ctW, nil, nil, cells, features, filters, classes, eval, params, sk)
				}
				
				ctBoot := cellCNN.DummyBootWithPrepool(ctTmp, 1, features, filters, classes, learningRate, momentum, params, sk)

				ctDC, ctDW, ctDCPrevBoot, ctDWPrevBoot = cellCNN.BackwardWithPrePooling(ctBoot, ptY, ptLBackward, features, filters, classes, j == batchSize-1, params, eval, sk)

				eval.Add(ctDC, ctDCPool, ctDCPool)
				eval.Add(ctDW, ctDWPool, ctDWPool)

				ctDCPool.SetScale(ctDC.Scale())
				ctDWPool.SetScale(ctDW.Scale())
				
				fmt.Printf("Sample[%d] : %s\n", k*batchSize+j, time.Since(start))
			}

			// Wt = avg(Wt)
			DCPool.MultConst(DCPool, complex(learningRate/float64(batchSize), 0))
			DWPool.MultConst(DWPool, complex(learningRate/float64(batchSize), 0))

			// W_i = learning_rate * Wt + W_i-1 * momentum
			DCPool.Add(DCPool, DCPrev)
			DWPool.Add(DWPool, DWPrev)

			DCPrev.MultConst(DCPool, complex(momentum, 0))
			DWPrev.MultConst(DWPool, complex(momentum, 0))

			// Subtstracks DC and DW to the weights
			C.Sub(C, DCPool)
			W.Sub(W, DWPool)

			// Masks DW and DC
			ctDCAvg := eval.MulNew(ctDCPool, maskPtCavg)
			ctDWAvg := eval.MulNew(ctDWPool, maskPtWavg)

			// Mask DWPrev*momentum and DCPrev*momentum
			eval.Mul(ctDCPrevBoot, maskPtC, ctDCPrevBoot)
			eval.Mul(ctDWPrevBoot, maskPtW, ctDWPrevBoot)

			// Adds DW with DWPrev*momentum 
			eval.Add(ctDCAvg, ctDCPrevBoot, ctDCAvg)
			eval.Add(ctDWAvg, ctDWPrevBoot, ctDWAvg)

			// Rescales
			eval.Rescale(ctDCAvg, params.Scale(), ctDCAvg)
			eval.Rescale(ctDWAvg, params.Scale(), ctDWAvg)

			// Stores DW + DWPrev*momentum 
			ctDCPrev = ctDCAvg.CopyNew().Ciphertext()
			ctDWPrev = ctDWAvg.CopyNew().Ciphertext()

			// Updates the weights
			eval.Sub(ctC, ctDCAvg, ctC)
			eval.Sub(ctW, ctDWAvg, ctW)

			cellCNN.DecryptPrintMatrix(C, true, ctC, params, sk)
			cellCNN.DecryptPrintMatrix(W.Transpose(), true, ctW, params, sk)
		}
	}
}


