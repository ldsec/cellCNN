package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/ldsec/cellCNN/cellCNNOptimized"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"runtime"
)

type Party struct {
	cellCNN.CellCNNProtocol
	XTrain []*cellCNN.Matrix
	YTrain []*cellCNN.Matrix
	prng   *cellCNN.PRNGInt
}

func NewParty(params ckks.Parameters) (p *Party) {
	p = new(Party)
	p.CellCNNProtocol = *cellCNN.NewCellCNNProtocol(params)
	return
}

func main() {

	hosts := cellCNN.Hosts

	trainEncrypted := false
	deterministic := true

	fmt.Printf("Loading Data... ")
	XTrain, YTrain := cellCNN.LoadTrainDataFrom("../../normalized2/", cellCNN.Samples, cellCNN.Cells, cellCNN.Features, cellCNN.Classes)
	XValid, YValid := cellCNN.LoadValidDataFrom("../../normalized2/", cellCNN.Samples, cellCNN.Cells, cellCNN.Features, cellCNN.Classes)
	fmt.Printf("Done\n")

	var C, W *cellCNN.Matrix
	if !deterministic {
		rand.Seed(time.Now().Unix())
		C = cellCNN.WeightsInit(cellCNN.Features, cellCNN.Filters, cellCNN.Features)
		W = cellCNN.WeightsInit(cellCNN.Filters, cellCNN.Classes, cellCNN.Filters)
	} else {
		C = new(cellCNN.Matrix)
		C.Rows = cellCNN.Features
		C.Cols = cellCNN.Filters
		C.Real = true
		C.M = cellCNN.C[:C.Rows*C.Cols]

		W = new(cellCNN.Matrix)
		W.Rows = cellCNN.Filters
		W.Cols = cellCNN.Classes
		W.Real = true
		W.M = cellCNN.W[:W.Rows*W.Cols]
	}

	params := cellCNN.GenParams()

	ringQP, _ := ring.NewRing(params.N(), append(params.Q(), params.P()...))

	P := make([]*Party, hosts)

	samplesPerHost := cellCNN.Samples / hosts

	for i := range P {

		P[i] = NewParty(params)
		P[i].SetWeights(C, W)

		if i-1 == hosts {
			P[i].XTrain = XTrain[i*samplesPerHost:]
			P[i].YTrain = YTrain[i*samplesPerHost:]

			samplesPerHost = (cellCNN.Samples / hosts) + (cellCNN.Samples % hosts)

		} else {
			P[i].XTrain = XTrain[i*samplesPerHost : (i+1)*samplesPerHost]
			P[i].YTrain = YTrain[i*samplesPerHost : (i+1)*samplesPerHost]
		}

		P[i].prng = cellCNN.NewPRNTInt(samplesPerHost, deterministic)
	}

	var masterSk *rlwe.SecretKey

	if trainEncrypted {

		fmt.Printf("Gen Keys... ")

		kgen := ckks.NewKeyGenerator(params)

		masterSk = ckks.NewSecretKey(params)

		for i := range P {
			ski := kgen.GenSecretKey()
			P[i].SetSecretKey(ski)
			ringQP.Add(masterSk.Value, P[i].Sk.Value, masterSk.Value)
		}

		pk := kgen.GenPublicKey(masterSk)
		rlk := kgen.GenRelinearizationKey(masterSk)

		rotations := P[0].RotKeyIndex()

		rtk := kgen.GenRotationKeysForRotations(rotations, true, masterSk)

		for i := range P {
			P[i].SetPublicKey(pk)
			P[i].EncryptWeights()
			P[i].EvaluatorInit(rlk, rtk)
		}

		fmt.Printf("Done\n")
	}

	slotUsage := 3*cellCNN.BatchSize*cellCNN.Filters*cellCNN.Classes + (2*cellCNN.Classes+1)*cellCNN.ConvolutionMatrixSize(cellCNN.BatchSize, cellCNN.Features, cellCNN.Filters)

	fmt.Printf("Slots Usage : %d/%d \n", slotUsage, params.Slots())

	epoch := 20
	niter := epoch * cellCNN.Samples / cellCNN.BatchSize

	fmt.Printf("#Iters : %d\n", niter)

	runtime.GC()

	for i := 0; i < niter; i++ {

		XPrePool := new(cellCNN.Matrix)
		XBatch := cellCNN.NewMatrix(cellCNN.BatchSize, cellCNN.Features)
		YBatch := cellCNN.NewMatrix(cellCNN.BatchSize, cellCNN.Classes)

		for j := range P {
			// Pre-pools the cells
			for k := 0; k < cellCNN.BatchSize; k++ {

				randi := P[j].prng.RandInt()

				X := P[j].XTrain[randi]
				Y := P[j].YTrain[randi]

				XPrePool.SumColumns(X)
				XPrePool.MultConst(XPrePool, complex(1.0/float64(cellCNN.Cells), 0))

				XBatch.SetRow(k, XPrePool.M)

				YBatch.SetRow(k, Y.M)
			}

			P[j].ForwardPlain(XBatch)
			P[j].BackWardPlain(XBatch, YBatch, hosts) // takes care of pre-applying 1/#Parties

			// === Ciphertext ===
			if trainEncrypted {

				start := time.Now()

				P[j].Forward(XBatch)
				P[j].Refresh(masterSk, hosts)
				P[j].Backward(XBatch, YBatch, hosts)

				/*
					fmt.Println("DC")
					P[j].DC.Print()
					cellCNN.DecryptPrint(cellCNN.Features, cellCNN.Filters, true, P[j].CtDC, params, masterSk)


					fmt.Println("DW")
					P[j].DW.Transpose().Print()
					for i := 0; i < cellCNN.Classes; i++{
						cellCNN.DecryptPrint(1, cellCNN.Filters, true, P[j].Eval.RotateNew(P[j].CtDW, i*cellCNN.BatchSize*cellCNN.Filters), params, masterSk)
					}
				*/

				fmt.Printf("Iter[%02d][%d] : %s\n", i, j, time.Since(start))
			}
		}

		// Aggregates the partial weights of all nodes
		for j := range P {
			if j != 0 {
				P[0].DC.Add(P[0].DC, P[j].DC)
				P[0].DW.Add(P[0].DW, P[j].DW)

				if trainEncrypted {
					P[0].Eval.Add(P[0].CtDC, P[j].CtDC, P[0].CtDC)
					P[0].Eval.Add(P[0].CtDW, P[j].CtDW, P[0].CtDW)
				}
			}
		}

		if trainEncrypted {
			fmt.Println("P[0] Aggregated DC Weights")
			P[0].DC.Print()
			cellCNN.DecryptPrint(cellCNN.Features, cellCNN.Filters, true, P[0].CtDC, params, masterSk)

			fmt.Println("P[0] Aggregated DW Weights")
			P[0].DW.Transpose().Print()
			for i := 0; i < cellCNN.Classes; i++ {
				cellCNN.DecryptPrint(1, cellCNN.Filters, true, P[0].Eval.RotateNew(P[0].CtDW, i*cellCNN.BatchSize*cellCNN.Filters), params, masterSk)
			}
		}

		// Updates local weights of the root node
		P[0].UpdatePlain()

		if trainEncrypted {
			P[0].Update()
		}

		// Copies the updated local weights of the root node on all the other nodes
		for j := range P {

			if j != 0 {
				copy(P[j].DC.M, P[0].DC.M)
				copy(P[j].DW.M, P[0].DW.M)

				if trainEncrypted {
					P[j].CtDC = P[0].CtDC.CopyNew()
					P[j].CtDW = P[0].CtDW.CopyNew()
				}
			}
		}

		runtime.GC()
	}

	if trainEncrypted {
		P[0].PrintCtWPrecision(masterSk)
		P[0].PrintCtCPrecision(masterSk)
	} else {
		P[0].C.Print()
		P[0].W.Print()
	}

	// Tests resuls :

	var encryptor ckks.Encryptor
	var decryptor ckks.Decryptor
	if trainEncrypted {
		encryptor = ckks.NewEncryptorFromSk(params, masterSk)
		decryptor = ckks.NewDecryptor(params, masterSk)
	}

	r := 0
	for i := 0; i < cellCNN.Samples/cellCNN.BatchSize; i++ {

		XPrePool := new(cellCNN.Matrix)
		XBatch := cellCNN.NewMatrix(cellCNN.BatchSize, cellCNN.Features)
		YBatch := cellCNN.NewMatrix(cellCNN.BatchSize, cellCNN.Classes)

		for j := 0; j < cellCNN.BatchSize; j++ {

			randi := rand.Intn(cellCNN.Samples)

			X := XValid[randi]
			Y := YValid[randi]

			XPrePool.SumColumns(X)
			XPrePool.MultConst(XPrePool, complex(1.0/float64(cellCNN.Cells), 0))

			XBatch.SetRow(j, XPrePool.M)
			YBatch.SetRow(j, Y.M)
		}

		v := P[0].PredictPlain(XBatch)

		if trainEncrypted {
			v.Print()

			ciphertexts := make([]*ckks.Ciphertext, cellCNN.Features>>1)
			for i := range ciphertexts {
				ciphertexts[i] = ckks.NewCiphertext(params, 1, 4, params.Scale())
			}

			cellCNN.EncryptLeftForCtMul(XBatch, cellCNN.Filters, 0.5, ciphertexts, P[0].Encoder, encryptor, params)

			ctPredict := cellCNN.Predict(ciphertexts, P[0].CtC, P[0].CtW, params, P[0].Eval)

			res := P[0].Encoder.Decode(decryptor.DecryptNew(ctPredict), params.LogSlots())

			U := cellCNN.NewMatrix(cellCNN.BatchSize, cellCNN.Classes)

			for i := 0; i < cellCNN.BatchSize; i++ {
				for j := 0; j < cellCNN.Classes; j++ {
					U.M[i*cellCNN.Classes+j] = res[i*cellCNN.Filters+cellCNN.BatchSize*cellCNN.Filters*j]
				}
			}

			U.Print()
			precisionStats := ckks.GetPrecisionStats(params, P[0].Encoder, nil, v.M, U.M, params.LogSlots(), 0)
			fmt.Printf("Batch[%2d]", i)
			fmt.Println(precisionStats.String())
		}

		for i := 0; i < cellCNN.BatchSize; i++ {

			idx := 0
			max := 0.0
			for j := 0; j < cellCNN.Classes; j++ {
				c := real(v.M[i*cellCNN.Classes+j])
				if c > max {
					idx = j
					max = c
				}
			}

			fmt.Println(i, v.M[i*cellCNN.Classes:(i+1)*cellCNN.Classes], YBatch.M[i*cellCNN.Classes:(i+1)*cellCNN.Classes])

			if int(real(YBatch.M[i*cellCNN.Classes+idx])) != 1 {
				r++
			}
		}
	}

	fmt.Printf("error : %v%s", 100.0*float64(r)/float64(cellCNN.Samples), "%")
}
