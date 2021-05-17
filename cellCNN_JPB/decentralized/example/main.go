package main

import (
	"fmt"
	"time"
	"math/rand"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/utils"
	"github.com/ldsec/cellCNN/cellCNN_JPB"
)


type Party struct{
	cellCNN.CellCNNProtocol
}

func NewParty(params *ckks.Parameters) (p *Party){
	p = new(Party)
	p.CellCNNProtocol = *cellCNN.NewCellCNNProtocol(params)
	return
}

func main() {

	var err error

	nParties := 3

	trainEncrypted := true
	deterministic := true

	if !deterministic{
		rand.Seed(time.Now().Unix())
	}
	
	params := cellCNN.GenParams()

	var prng utils.PRNG
	if prng, err = utils.NewPRNG(); err != nil {
		panic(err)
	}

	ringQP, _ := ring.NewRing(params.N(), append(params.Qi(), params.Pi()...))

	crpGenerator := ring.NewUniformSampler(prng, ringQP)

	fmt.Println(params.LogQP())

	// GlobalWeights
	C := cellCNN.WeightsInit(cellCNN.Features, cellCNN.Filters, cellCNN.Features)
	W := cellCNN.WeightsInit(cellCNN.Filters, cellCNN.Classes, cellCNN.Filters) 


	P := make([]*Party, nParties)

	for i := range P{
		P[i] = NewParty(params)
		P[i].SetWeights(C, W)
	}

	GenEncryptionKey(P, crpGenerator)

	masterSk := ckks.NewSecretKey(params)

	for i := range P{
		ringQP.Add(masterSk.SecretKey.Value, P[i].SK().SecretKey.Value, masterSk.SecretKey.Value)
	}

	fmt.Printf("Weights Encryption... ")
	for i := range P{
		P[i].EncryptWeights()
	}
	fmt.Printf("Done\n")

	GenRelinearizationKey(P, crpGenerator, params)
	GenRotationKeys(P, crpGenerator, params)

	for i := range P{
		P[i].EvaluatorInit()
	}
	fmt.Printf("Done\n")

	P[0].Eval().Rotate(P[0].CtC(), 1, P[0].CtC())

	fmt.Println("DC")
	P[0].C.Print()
	pt := CollectiveDecryption(P, P[0].CtC(), params)
	cellCNN.DecryptPrint(cellCNN.Features, cellCNN.Filters, true, pt, params, nil)

	

	slotUsage := 3*cellCNN.BatchSize*cellCNN.Filters*cellCNN.Classes + (2*cellCNN.Classes+1) * cellCNN.ConvolutionMatrixSize(cellCNN.BatchSize, cellCNN.Features, cellCNN.Filters)

	fmt.Printf("Slots Usage : %d/%d \n", slotUsage, params.Slots()) 


	fmt.Println("Loading Data ...")
	XTrain, YTrain := cellCNN.LoadTrainDataFrom("../normalized/", 2000, cellCNN.Cells, cellCNN.Features)
	XValid, YValid := cellCNN.LoadValidDataFrom("../normalized/", 2000, cellCNN.Cells, cellCNN.Features)
	fmt.Println("Done")

	epoch := 1
	niter := epoch * cellCNN.Samples / cellCNN.BatchSize

	fmt.Printf("#Iters : %d\n", niter)

	partyDataSize := 2000/nParties

	for i := 0; i < niter; i++{

		XPrePool := new(ckks.Matrix)
		XBatch := ckks.NewMatrix(cellCNN.BatchSize, cellCNN.Features)
		YBatch := ckks.NewMatrix(cellCNN.BatchSize, cellCNN.Classes)

		DWPool := ckks.NewMatrix(cellCNN.Filters, cellCNN.Classes)
		DCPool := ckks.NewMatrix(cellCNN.Features, cellCNN.Filters)

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
			P[j].BackWardPlain(XBatch, YBatch)

			DWPool.Add(DWPool, P[j].DW)
			DCPool.Add(DCPool, P[j].DC)

			// === Ciphertext === 
			if trainEncrypted{

				start := time.Now()

				P[j].Forward(XBatch)
				P[j].Refresh()
				P[j].Backward(XBatch, YBatch)

				fmt.Println("DC")
				P[j].DC.Print()
				//pt := CollectiveDecryption(P, P[j].CtDC(), params)
				cellCNN.DecryptPrint(cellCNN.Features, cellCNN.Filters, true, P[j].CtDC(), params, masterSk)

				fmt.Println("DW")
				P[j].DW.Transpose().Print()
				for i := 0; i < cellCNN.Classes; i++{
					//pt := CollectiveDecryption(P, P[j].Eval().RotateNew(P[j].CtDW(), i*cellCNN.BatchSize*cellCNN.Filters), params)
					cellCNN.DecryptPrint(1, cellCNN.Filters, true, P[j].Eval().RotateNew(P[j].CtDW(), i*cellCNN.BatchSize*cellCNN.Filters), params, masterSk)
				}

				fmt.Printf("Iter[%02d] : %s\n", i, time.Since(start))

				if j == 0{
					ctDWPool = P[j].CtDW().CopyNew().Ciphertext()
					ctDCPool = P[j].CtDC().CopyNew().Ciphertext()
				}else{
					P[0].Eval().Add(ctDWPool, P[j].CtDW(), ctDWPool)
					P[0].Eval().Add(ctDCPool, P[j].CtDC(), ctDCPool)
				}
			}
		}

		DWPool.MultConst(DWPool, 1/complex(float64(nParties), 0))
		DCPool.MultConst(DCPool, 1/complex(float64(nParties), 0))

		for j := range P{
			P[j].UpdatePlain(DCPool, DWPool)
			P[j].Update(ctDCPool, ctDWPool)
		}

	}

	for i := range P{
		P[i].W.Transpose().Print()
	}

	//cellCNNProtocol.PrintCtWPrecision()
	//cellCNNProtocol.PrintCtCPrecision()

	// Tests resuls :
 
	r := 0
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

		v := P[0].PredictPlain(XBatch)

		/*
		if trainEncrypted {
			v.Print()
			ctv := cellCNNProtocol.Predict(XBatch)
			ctv.Print()
			precisionStats := ckks.GetPrecisionStats(params, cellCNNProtocol.Encoder(), nil, v.M, ctv.M, 0)
			fmt.Printf("Batch[%2d]", i)
			fmt.Println(precisionStats.String())
		}
		*/
		
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

func CollectiveDecryption(P []*Party, ciphertext *ckks.Ciphertext, params *ckks.Parameters) (*ckks.Plaintext){
	for i := range P{
		P[i].NewCKSProtocol()
	}


	zero := params.NewPolyQ()
	for i := range P{
		P[i]. CKSGenShare(ciphertext, zero)
	}

	for i := range P[1:]{
		P[0].CKSAggregate(P[i+1].CKSGetShare())
	}

	return P[0].CKSKeySwitchToPlaintext(ciphertext)
}

func GenRotationKeys(P []*Party, crpGenerator *ring.UniformSampler, params *ckks.Parameters){

	fmt.Printf("Generating Rotation Keys... ")

	for i := range P{
		P[i].NewRTGProtocol()
	}

	var crpRTG []*ring.Poly
	var galEl uint64

	// Rotations Keys
	for _, rot := range P[0].RotKeyIndex(){

		galEl = params.GaloisElementForColumnRotationBy(rot)

		if !P[0].HasRotKey(galEl) && rot != 0{

			fmt.Printf("%d ", rot)

			if crpRTG == nil {

				crpRTG = make([]*ring.Poly, params.Beta())

				for i := 0; i < params.Beta(); i++ {
					crpRTG[i] = crpGenerator.ReadNew()
				}

			}else{
				for i := 0; i < params.Beta(); i++ {
					crpGenerator.Read(crpRTG[i])
				}
			}

			for i := range P{
				P[i].RTGGenShare(galEl, crpRTG)
			}

			for i := range P{
				for j := range P{
					if i != j {
						P[i].RTGAggregate(P[j].RTGGetShare())
					}
				}
			}

			for i := range P{
				P[i].RTGGenRotationKey(galEl, crpRTG)
			}
		}
	}

	// Conjugate Key
	for i := 0; i < params.Beta(); i++ {
		crpGenerator.Read(crpRTG[i])
	}

	galEl = params.GaloisElementForRowRotation()
		
	for i := range P{
		P[i].RTGGenShare(galEl, crpRTG)
	}

	for i := range P{
		for j := range P{
			if i != j {
				P[i].RTGAggregate(P[j].RTGGetShare())
			}
		}
	}

	for i := range P{
		P[i].RTGGenRotationKey(galEl, crpRTG)
		P[i].RTGWipe()
	}

	fmt.Printf("Done\n")
}

func GenRelinearizationKey(P []*Party, crpGenerator *ring.UniformSampler, params *ckks.Parameters){

	fmt.Printf("Generating Relinearization Key... ")

	crpRKG := make([]*ring.Poly, params.Beta())

	for i := 0; i < params.Beta(); i++ {
		crpRKG[i] = crpGenerator.ReadNew()
	}

	for i := range P{
		P[i].NewRKGProtocol()
	}

	for i := range P{
		P[i].RKGRoundOne(crpRKG)
	}

	for i := range P{
		for j := range P{
			if i != j {
				P[i].CKGAggregateRoundOne(P[j].CKGGetShareOne())
			}
		}
	}

	for i := range P{
		P[i].RKGRoundTwo(crpRKG)
	}

	for i := range P{
		for j := range P{
			if i != j {
				P[i].CKGAggregateRoundTwo(P[j].CKGGetShareTwo())
			}
		}
	}

	for i := range P{
		P[i].RKGGenRelinearizationKey()
		P[i].CKGWipe()
	}

	fmt.Printf("Done\n")
}


func GenEncryptionKey(P []*Party, crpGenerator *ring.UniformSampler){
	fmt.Printf("Generating Encryption Key... ")

	crpCKG := crpGenerator.ReadNew()

	for i := range P{
		P[i].NewCKGProtocol()
	}

	for i := range P{
		P[i].CKGGenShare(crpCKG)
	}

	for i := range P{
		for j := range P{
			if i != j {
				P[i].CKGAggregateShares(P[j].CKGGetShare())
			}
		}
	}

	for i := range P{
		P[i].CKGGenPublicKey(crpCKG)
		P[i].CKGWipe()
	}

	fmt.Printf("Done\n")
}