package main

import (
	"fmt"
	"time"
	"math/rand"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/cellCNN/cellCNN_JPB"
)




func main() {

	trainEncrypted := true
	deterministic := true

	if !deterministic{
		rand.Seed(time.Now().Unix())
	}
	
	params := cellCNN.GenParams()

	fmt.Println(params.LogQP())

	// Keys
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()

	batchSize := cellCNN.BatchSize
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


	fmt.Println("Loading Data ...")
	XTrain, YTrain := cellCNN.LoadTrainDataFrom("../normalized/", 2000, cellCNN.Cells, cellCNN.Features)
	XValid, YValid := cellCNN.LoadValidDataFrom("../normalized/", 2000, cellCNN.Cells, cellCNN.Features)
	fmt.Println("Done")

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


	cellCNNProtocol := cellCNN.NewCellCNNProtocol(params, sk)

	C, W := cellCNNProtocol.C, cellCNNProtocol.W

	epoch := 15
	niter := epoch * samples / batchSize

	niter = 4

	fmt.Printf("#Iters : %d\n", niter)

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

			start := time.Now()

			cellCNNProtocol.Forward(XBatch)
			cellCNNProtocol.Refresh()
			cellCNNProtocol.Backward(YBatch, XBatch.Transpose())
			cellCNNProtocol.Update()

			fmt.Printf("Iter[%02d] : %s\n", i, time.Since(start))
		}
	}

	ctDC, ctDW := cellCNNProtocol.UpdatedWeights()

	// Visual comparison between plaintext and ciphertext model (should match ~8 decimal digits)
	if trainEncrypted {
		fmt.Println("DC")
		DCBatch.Print()
		cellCNN.DecryptPrint(features, filters, true, ctDC, params, sk)

		fmt.Println("DW")
		DWBatch.Transpose().Print()
		for i := 0; i < classes; i++{
			cellCNN.DecryptPrint(1, filters, true, cellCNNProtocol.Eval().RotateNew(ctDW, i*batchSize*filters), params, sk)
		}
	}

	// Tests resuls :

	err := 0
	var v int
	nTests := 2000
	for i := 0; i < nTests; i++{
		X := XValid[i]
		Y := YValid[i]

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


