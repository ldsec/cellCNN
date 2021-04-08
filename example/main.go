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

	batchSize := cellCNN.BatcheSize
	samples := cellCNN.Samples
	cells := cellCNN.Cells
	features := cellCNN.Features
	filters := cellCNN.Filters
	classes := cellCNN.Classes

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

	// Repacking of ctPpool before bootstrapping
	rotations = append(rotations, -((cells-1)*filters+classes*filters))

	// Repacking of ctW before bootstrapping
	rotations = append(rotations, -(2*(cells-1)*filters + 2*classes*filters))
	rotations = append(rotations, classes*filters*features + classes*filters)

	rotations = append(rotations, features*filters)
	rotations = append(rotations, -features*filters)

	rotations = append(rotations, classes*filters + classes*filters*features + classes * filters)

	rotations = append(rotations, classes*(cells*filters + filters + features*filters) + classes*filters)

	rotations = append(rotations, -(classes * filters + 2*(cells-1)*filters + 2*classes*filters))
	rotations = append(rotations, -(classes * filters + 2*(cells-1)*filters + 3*classes*filters))

	rotations = append(rotations, 2*(classes*filters + classes * filters * (cells + features + 1)))
	rotations = append(rotations, 2*(classes*filters + classes * filters * (cells + features + 1)) + classes*filters)


	rotkeys := kgen.GenRotationKeysForRotations(rotations, true, sk)

	eval := ckks.NewEvaluator(params, ckks.EvaluationKey{rlk, rotkeys})
	_=eval

	XTrain, YTrain := cellCNN.LoadTrainDataFrom("../normalized/", samples, cellCNN.Cells, cellCNN.Features)
	C := cellCNN.WeightsInit(features, filters, features)
	W := cellCNN.WeightsInit(filters, classes, filters) 

	ctC := cellCNN.EncryptRightForPtMul(C, cells, params, levelC, sk)
	_=ctC
	// Returns
	//
	// [[ W transpose row encoded ] [         available         ]]
	//  |    classes * filters    | | Slots - classes * filters | 
	//
	ctW := cellCNN.EncryptRightForNaiveMul(W, params, levelW, sk)
	_=ctW

	P := new(ckks.Matrix)
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
	DCPool := ckks.NewMatrix(features, filters)
	DWPool := ckks.NewMatrix(filters, classes)

	maskW := make([]complex128, params.Slots())

	for i := 0; i < classes*filters; i++ {
		maskW[i] = complex(1, 0)
	}

	encoder := ckks.NewEncoder(params)
	maskPtW := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	encoder.EncodeNTT(maskPtW, maskW, params.LogSlots())


	maskC := make([]complex128, params.Slots())

	for i := 0; i < classes * filters * (cells + features + 1); i++ {
		maskC[i] = complex(1, 0)
	}

	maskPtC := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	encoder.EncodeNTT(maskPtC, maskC, params.LogSlots())

	epoch := 30

	//var ctDW, ctDC *ckks.Ciphertext

	for i := 0; i < epoch; i++{

		for k := 0; k < samples/batchSize; k++ {

			fmt.Printf("Epoch[%02d] | Batch[%d]\n", i, k)

			for j := 0; j < samples; j++ {

				//ptL := cellCNN.EncodeLeftForPtMul(XTrain[j], filters, 1, params)
				//ptY := cellCNN.EncodeLabelsForBackward(YTrain[j], cells,features, filters, classes, params)
				//ptLBackward := cellCNN.EncodeCellsForBackward(XTrain[j], cells, features, filters, classes, learningRate, params)

				// =======================================
				// ========== Plaintext circuit ==========
				// =======================================

				X := XTrain[k*batchSize + j]
				Y := YTrain[k*batchSize + j]

				P.MulMat(X, C)

				Ppool.SumColumns(P)
				Ppool.MultConst(Ppool, complex(1.0/float64(cells), 0))

				U.MulMat(Ppool, W)

				L1.Func(U, cellCNN.Activation)
				L1Deriv.Func(U, cellCNN.ActivationDeriv)

				E1.Sub(Y, L1)
				E1.Dot(E1, L1Deriv)

				DW.MulMat(Ppool.Transpose(), E1)

				E0.MulMat(E1, W.Transpose())

				DC.SumRows(X.Transpose())
				DC.MultConst(DC, complex(1.0/float64(cells), 0))

				DC.MulMat(DC, E0)

				// Pools the gradients of the batch
				DCPool.Add(DCPool, DC)
				DWPool.Add(DWPool, DW)
				
				// =======================================

				
				
				start := time.Now()
				_=start
				/*

				ctTmp := cellCNN.Forward(ptL, ctC, ctW, ctDW, ctDC, cells, features, filters, classes, eval, params, sk)

				ctBoot := cellCNN.DummyBoot(ctTmp, cells, features, filters, classes, learningRate, momentum, params, sk)

				ctDW, ctDC = cellCNN.Backward(ctBoot, ptY, ptLBackward, maskPtW, maskPtC, cells, features, filters, classes, params, eval, sk)

				eval.Sub(ctW, ctDW, ctW)
				eval.Sub(ctC, ctDC, ctC)
				
				fmt.Printf("Total : %s\n", time.Since(start))
				fmt.Println()



				fmt.Println("DW")
				DWt := DW.Transpose()
				DWt.Print()
				cellCNN.DecryptPrintMatrix(classes, filters, true, ctDW, params, sk)

				fmt.Println("DC")
				DC.Print()
				cellCNN.DecryptPrintMatrix(features, filters, true, ctDC, params, sk)
				*/
			}

			
			// Multiplies by learning rate and takes the average of sum(DW) and sum(DC)
			DCPool.MultConst(DCPool, complex(learningRate/float64(batchSize), 0))
			DWPool.MultConst(DWPool, complex(learningRate/float64(batchSize), 0))

			// Adds the previous DW and DC
			DCPool.Add(DCPool, DCPrev)
			DWPool.Add(DWPool, DWPrev)

			// Stores the current DW and DC and multiplies by the momentum
			DCPrev.MultConst(DCPool, complex(momentum, 0))
			DWPrev.MultConst(DWPool, complex(momentum, 0))

			// Subtstracks DC and DW to the weights
			C.Sub(C, DCPool)
			W.Sub(W, DWPool)
			
		}
	}
}


