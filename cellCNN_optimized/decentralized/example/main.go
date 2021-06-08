package main

import (
	"fmt"
	"time"
	"math/rand"

	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/cellCNN/cellCNN_optimized"
	"runtime"
)


type Party struct{
	cellCNN.CellCNNProtocol
}

func NewParty(params ckks.Parameters) (p *Party){
	p = new(Party)
	p.CellCNNProtocol = *cellCNN.NewCellCNNProtocol(params)
	return
}

func main() {

	nParties := 1

	trainEncrypted := true
	deterministic := true

	if !deterministic{
		rand.Seed(time.Now().Unix())
	}
	
	params := cellCNN.GenParams()

	ringQP, _ := ring.NewRing(params.N(), append(params.Q(), params.P()...))

	fmt.Println(params.LogQP())

	// GlobalWeights
	C := cellCNN.WeightsInit(cellCNN.Features, cellCNN.Filters, cellCNN.Features)
	W := cellCNN.WeightsInit(cellCNN.Filters, cellCNN.Classes, cellCNN.Filters) 


	P := make([]*Party, nParties)

	for i := range P{
		P[i] = NewParty(params)
		P[i].SetWeights(C, W)
	}

	var masterSk *rlwe.SecretKey

	if trainEncrypted {

		kgen := ckks.NewKeyGenerator(params)
		
		masterSk = ckks.NewSecretKey(params)

		for i := range P{
			ski := kgen.GenSecretKey()
			P[i].SetSecretKey(ski)
			ringQP.Add(masterSk.Value, P[i].SK().Value, masterSk.Value)
		}

		pk := kgen.GenPublicKey(masterSk)
		rlk := kgen.GenRelinearizationKey(masterSk)

		rotations := P[0].RotKeyIndex()

		rtk := kgen.GenRotationKeysForRotations(rotations, true, masterSk)

		for i := range P{
			P[i].SetPublicKey(pk)
			P[i].EncryptWeights()
			P[i].EvaluatorInit(rlk, rtk)
		}
	}

	slotUsage := 3*cellCNN.BatchSize*cellCNN.Filters*cellCNN.Classes + (2*cellCNN.Classes+1) * cellCNN.ConvolutionMatrixSize(cellCNN.BatchSize, cellCNN.Features, cellCNN.Filters)

	fmt.Printf("Slots Usage : %d/%d \n", slotUsage, params.Slots()) 


	fmt.Println("Loading Data ...")
	XTrain, YTrain := cellCNN.LoadTrainDataFrom("../normalized/", 2000, cellCNN.Cells, cellCNN.Features)
	XValid, YValid := cellCNN.LoadValidDataFrom("../normalized/", 2000, cellCNN.Cells, cellCNN.Features)
	fmt.Println("Done")

	epoch := 15
	niter := epoch * cellCNN.Samples / cellCNN.BatchSize

	niter = 1

	fmt.Printf("#Iters : %d\n", niter)

	partyDataSize := 2000/nParties

	runtime.GC()

	for i := 0; i < niter; i++{

		XPrePool := new(cellCNN.Matrix)
		XBatch := cellCNN.NewMatrix(cellCNN.BatchSize, cellCNN.Features)
		YBatch := cellCNN.NewMatrix(cellCNN.BatchSize, cellCNN.Classes)

		DWPool := cellCNN.NewMatrix(cellCNN.Filters, cellCNN.Classes)
		DCPool := cellCNN.NewMatrix(cellCNN.Features, cellCNN.Filters)

		var ctDWPool, ctDCPool *ckks.Ciphertext

		for j := range P{
			// Pre-pools the cells
			for k := 0; k < cellCNN.BatchSize; k++ {

				partyBatchStart := j*partyDataSize
				partyBatchEnd := (j+1)*partyDataSize

				randi := rand.Intn(partyDataSize)

				X := XTrain[partyBatchStart:partyBatchEnd][randi]
				Y := YTrain[partyBatchStart:partyBatchEnd][randi]

				XPrePool.SumColumns(X)
				XPrePool.MultConst(XPrePool, complex(1.0/float64(cellCNN.Cells), 0))

				XBatch.SetRow(k, XPrePool.M)
				YBatch.SetRow(k, []complex128{Y.M[1], Y.M[0]})
			}

			P[j].ForwardPlain(XBatch)
			P[j].BackWardPlain(XBatch, YBatch, nParties) // takes care of pre-applying 1/#Parties

			DWPool.Add(DWPool, P[j].DW)
			DCPool.Add(DCPool, P[j].DC)

			// === Ciphertext === 
			if trainEncrypted{

				start := time.Now()

				P[j].Forward(XBatch)
				P[j].Refresh(masterSk, P[j].CtBoot(), nParties)
				P[j].Backward(XBatch, YBatch, nParties)


				/*
				fmt.Println("DC")
				P[j].DC.Print()
				cellCNN.DecryptPrint(cellCNN.Features, cellCNN.Filters, true, P[j].CtDC(), params, masterSk)

				fmt.Println("DW")
				P[j].DW.Transpose().Print()
				for i := 0; i < cellCNN.Classes; i++{
					cellCNN.DecryptPrint(1, cellCNN.Filters, true, P[j].Eval().RotateNew(P[j].CtDW(), i*cellCNN.BatchSize*cellCNN.Filters), params, masterSk)
				}
				*/

				fmt.Printf("Iter[%02d][%d] : %s\n", i, j, time.Since(start))

				if j == 0{
					ctDWPool = P[j].CtDW().CopyNew()
					ctDCPool = P[j].CtDC().CopyNew()
				}else{
					P[0].Eval().Add(ctDWPool, P[j].CtDW(), ctDWPool)
					P[0].Eval().Add(ctDCPool, P[j].CtDC(), ctDCPool)
				}
			}
		}


		for j := range P{
			P[j].UpdatePlain(DCPool, DWPool)

			if trainEncrypted{
				P[j].Update(ctDCPool, ctDWPool)
			}
		}

		if trainEncrypted{
			fmt.Println("DCPool")
			DCPool.Print()
			//pt := CollectiveDecryption(P, P[j].CtDC(), params)
			cellCNN.DecryptPrint(cellCNN.Features, cellCNN.Filters, true, ctDCPool, params, masterSk)

			fmt.Println("DWPool")
			DWPool.Transpose().Print()
			for i := 0; i < cellCNN.Classes; i++{
				//pt := CollectiveDecryption(P, P[j].Eval().RotateNew(P[j].CtDW(), i*cellCNN.BatchSize*cellCNN.Filters), params)
				cellCNN.DecryptPrint(1, cellCNN.Filters, true, P[0].Eval().RotateNew(ctDWPool, i*cellCNN.BatchSize*cellCNN.Filters), params, masterSk)
			}
		}

		runtime.GC()
	}

	if trainEncrypted {
		P[0].PrintCtWPrecision(masterSk)
		P[0].PrintCtCPrecision(masterSk)
	}
	

	// Tests resuls :
 
	r := 0
	for i := 0; i < 2000/cellCNN.BatchSize; i++{

		XPrePool := new(cellCNN.Matrix)
		XBatch := cellCNN.NewMatrix(cellCNN.BatchSize, cellCNN.Features)
		YBatch := cellCNN.NewMatrix(cellCNN.BatchSize, cellCNN.Classes)

		for j := 0; j < cellCNN.BatchSize; j++ {

			randi := rand.Intn(2000)

			X := XValid[randi]
			Y := YValid[randi]

			XPrePool.SumColumns(X)
			XPrePool.MultConst(XPrePool, complex(1.0/float64(cellCNN.Cells), 0))

			XBatch.SetRow(j, XPrePool.M)
			YBatch.SetRow(j, []complex128{Y.M[1], Y.M[0]})
		}

		v := P[0].PredictPlain(XBatch)

		if trainEncrypted {
			v.Print()
			ctv := P[0].Predict(XBatch, masterSk)
			ctv.Print()
			precisionStats := ckks.GetPrecisionStats(params, P[0].Encoder(), nil, v.M, ctv.M, params.LogSlots(), 0)
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
				r++
			}
		}
	}

	fmt.Printf("error : %v%s", 100.0*float64(r)/float64(2000), "%")
}