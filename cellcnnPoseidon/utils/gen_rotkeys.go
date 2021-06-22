package utils

import (
	"fmt"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
)

// InitEvaluator init evaluator for the distributed training
// same function for cellcnn.InitEvaluator,for special use in decentralized settings
func InitEvaluator(cp *CryptoParams, sts *CellCnnSettings, maxM1N2Ratio float64) (ckks.Evaluator, *ckks.PtDiagMatrix) {
	kgen := cp.Kgen()
	aggSk := cp.AggregateSk
	relikey := cp.Rlk

	t1 := time.Now()

	Cinds := InitConvRotationInds(sts, cp.Params)
	Dinds, diagM := InitDenseRotationInds(sts, kgen, cp.Params, cp.GetEncoder(), maxM1N2Ratio)
	// clear duplicate
	Rinds := ClearRotInds(append(Cinds, Dinds...), cp.Params.Slots())

	rks := kgen.GenRotationKeysForRotations(Rinds, false, aggSk)

	t2 := time.Since(t1).Seconds()

	eval := ckks.NewEvaluator(cp.Params, rlwe.EvaluationKey{Rlk: relikey, Rtks: rks})

	fmt.Printf("==> Init evaluators for all nodes: %v s\n", t2)

	return eval, diagM
}

// InitConvRotationInds rotation keys for conv
func InitConvRotationInds(sts *CellCnnSettings, params ckks.Parameters) []int {
	nmakers := sts.Nmakers
	nfilters := sts.Nfilters
	ncells := sts.Ncells

	// Conv1D Forward
	// 1. for input weights matrix mult
	Fmult := params.RotationsForInnerSumLog(1, nmakers*ncells)

	// 2. for rotation the result to left most slots
	Fshift := make([]int, 0)
	for i := 1; i < nfilters; i++ {
		Fshift = append(Fshift, -i)
	}
	Finds := append(Fmult, Fshift...)

	//Conv1D Backward
	// 3. for replicate the err for each filter
	Brep1 := NewSlice(0, (sts.Nfilters-1)*sts.Nclasses, sts.Nclasses)
	Brep2 := params.RotationsForInnerSumLog(-1, sts.Ncells*sts.Nmakers)
	Brep := append(Brep1, Brep2...)

	// 4. left collect gradient
	Bcol := make([]int, sts.Nmakers)
	for i := 0; i < sts.Nmakers; i++ {
		Bcol[i] = sts.Ncells*i - i
	}
	// 5. replicate gradient
	Brepg := params.RotationsForInnerSumLog(-sts.Nmakers, sts.Ncells)
	B0 := params.RotationsForInnerSumLog(1, sts.Ncells)
	Bcollect := params.RotationsForInnerSumLog(sts.Ncells-1, sts.Nmakers)

	Binds := append(Brep, Bcol...)
	Binds = append(Binds, Brepg...)
	Binds = append(Binds, B0...)
	Binds = append(Binds, Bcollect...)

	Inds := append(Finds, Binds...)

	return Inds
}

// InitDenseRotationInds rotation keys for dense
func InitDenseRotationInds(sts *CellCnnSettings, kgen ckks.KeyGenerator, params ckks.Parameters, encoder ckks.Encoder, maxM1N2Ratio float64) ([]int, *ckks.PtDiagMatrix) {
	nfilters := sts.Nfilters
	nclasses := sts.Nclasses

	// Dense Forward
	// 1. for replicate the input mutiple times
	Frep := params.RotationsForInnerSumLog(-sts.Nfilters, sts.Nclasses)

	// 2. for input weights matrix mult
	Fmult := params.RotationsForInnerSumLog(1, nfilters)

	// 3. for collect the result into the left most slots
	Finds := append(Frep, Fmult...)

	// Dense Backward
	rot1 := params.RotationsForInnerSumLog(-1, sts.Nfilters)
	rot2 := params.RotationsForInnerSumLog(-sts.Nfilters, sts.Nclasses)
	rot3 := params.RotationsForInnerSumLog(-sts.Nclasses, sts.Nfilters)
	rotAll := append(rot1, rot2...)
	rotAll = append(rotAll, rot3...)

	// 	transpose
	inRowPacked := false
	ouRowPacked := false
	colsMatrix := GenTransposeMatrix(params.Slots(), nfilters, nclasses, inRowPacked, ouRowPacked)
	transposeVec := GenTransposeMap(colsMatrix)
	diagM := encoder.EncodeDiagMatrixBSGSAtLvl(params.MaxLevel(), transposeVec, params.Scale(), maxM1N2Ratio, params.LogSlots())

	Btranspose := params.RotationsForDiagMatrixMult(diagM)
	Binds := append(rotAll, Btranspose...)

	return append(Finds, Binds...), diagM
}
