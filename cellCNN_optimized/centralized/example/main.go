package main

import (
	"fmt"
	"time"
	"math/rand"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/cellCNN/cellCNN_JPB"
)


type Party struct{
	cellCNN.CellCNNProtocol
}

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

	slotUsage := 3*cellCNN.BatchSize*cellCNN.Filters*cellCNN.Classes + (2*cellCNN.Classes+1) * cellCNN.ConvolutionMatrixSize(cellCNN.BatchSize, cellCNN.Features, cellCNN.Filters)

	fmt.Printf("Slots Usage : %d/%d \n", slotUsage, params.Slots()) 


	fmt.Println("Loading Data ...")
	XTrain, YTrain := cellCNN.LoadTrainDataFrom("../normalized/", 2000, cellCNN.Cells, cellCNN.Features)
	XValid, YValid := cellCNN.LoadValidDataFrom("../normalized/", 2000, cellCNN.Cells, cellCNN.Features)
	fmt.Println("Done")

	fmt.Println("Generating Keys")
	cellCNNProtocol := cellCNN.NewCellCNNProtocol(params, sk)
	fmt.Println("Done")

	epoch := 15
	niter := epoch * cellCNN.Samples / cellCNN.BatchSize

	fmt.Printf("#Iters : %d\n", niter)

	for i := 0; i < niter; i++{

		XPrePool := new(ckks.Matrix)
		XBatch := ckks.NewMatrix(cellCNN.BatchSize, cellCNN.Features)
		YBatch := ckks.NewMatrix(cellCNN.BatchSize, cellCNN.Classes)

		// Pre-pools the cells
		for j := 0; j < cellCNN.BatchSize; j++ {

			randi := rand.Intn(2000)

			X := XTrain[randi]
			Y := YTrain[randi]

			XPrePool.SumColumns(X)
			XPrePool.MultConst(XPrePool, complex(1.0/float64(cellCNN.Cells), 0))

			XBatch.SetRow(j, XPrePool.M)
			YBatch.SetRow(j, []complex128{Y.M[1], Y.M[0]})
		}

		// === Plaintext ===

		cellCNNProtocol.ForwardPlain(XBatch)
		cellCNNProtocol.BackWardPlain(XBatch, YBatch)
		cellCNNProtocol.UpdatePlain()

		// === Ciphertext === 
		if trainEncrypted{

			start := time.Now()

			cellCNNProtocol.Forward(XBatch)
			cellCNNProtocol.Refresh()
			cellCNNProtocol.Backward(XBatch, YBatch)


			/*
			fmt.Println("DC")
			cellCNNProtocol.DC.Print()
			cellCNN.DecryptPrint(cellCNN.Features, cellCNN.Filters, true, cellCNNProtocol.CtDC(), params, sk)

			fmt.Println("DW")
			cellCNNProtocol.DW.Transpose().Print()
			for i := 0; i < cellCNN.Classes; i++{
				cellCNN.DecryptPrint(1, cellCNN.Filters, true, cellCNNProtocol.Eval().RotateNew(cellCNNProtocol.CtDW(), i*cellCNN.BatchSize*cellCNN.Filters), params, sk)
			}
			*/

			cellCNNProtocol.Update(cellCNNProtocol.CtDC(), cellCNNProtocol.CtDW())
			fmt.Printf("Iter[%02d] : %s\n", i, time.Since(start))
		}
	}

	cellCNNProtocol.PrintCtWPrecision()
	cellCNNProtocol.PrintCtCPrecision()

	// Tests resuls :
 
	err := 0
	for i := 0; i < 2000/cellCNN.BatchSize; i++{

		XPrePool := new(ckks.Matrix)
		XBatch := ckks.NewMatrix(cellCNN.BatchSize, cellCNN.Features)
		YBatch := ckks.NewMatrix(cellCNN.BatchSize, cellCNN.Classes)

		for j := 0; j < cellCNN.BatchSize; j++ {

			randi := rand.Intn(2000)

			X := XValid[randi]
			Y := YValid[randi]

			XPrePool.SumColumns(X)
			XPrePool.MultConst(XPrePool, complex(1.0/float64(cellCNN.Cells), 0))

			XBatch.SetRow(j, XPrePool.M)
			YBatch.SetRow(j, []complex128{Y.M[1], Y.M[0]})
		}

		v := cellCNNProtocol.PredictPlain(XBatch)

		if trainEncrypted {
			v.Print()
			ctv := cellCNNProtocol.Predict(XBatch)
			ctv.Print()
			precisionStats := ckks.GetPrecisionStats(params, cellCNNProtocol.Encoder(), nil, v.M, ctv.M, 0)
			fmt.Printf("Batch[%2d]", i)
			fmt.Println(precisionStats.String())
		}
		
		var y int
		for i := 0; i < cellCNN.BatchSize; i++{

			if real(v.M[i*2]) > real(v.M[i*2+1]){
				y = 1
			}else{
				y = 0
			}

			if y != int(real(YBatch.M[i*2])){
				err++
			}
		}
	}

	fmt.Println("error : ", float64(err)/float64(2000))

}


