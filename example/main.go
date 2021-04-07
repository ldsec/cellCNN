package main

import (
	"fmt"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/cellCNN"
)




func main() {

	params := cellCNN.GenParams(float64(1 << 52))

	fmt.Println(params.LogQP())

	//encoder := ckks.NewEncoder(params)

	// Keys
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()

	// Relinearization key
	rlk := kgen.GenRelinearizationKey(sk)

	cells := cellCNN.Cells
	features := cellCNN.Features
	filters := cellCNN.Filters
	classes := cellCNN.Classes

	learningRate := cellCNN.LearningRate

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
	
	//rotations = append(rotations, GenReplicateRotKeyIndex(features*filters, int(float64(cells)/float64(features) + 1.5))...)

	rotkeys := kgen.GenRotationKeysForRotations(rotations, true, sk)

	eval := ckks.NewEvaluator(params, ckks.EvaluationKey{rlk, rotkeys})

	

	L := ckks.GenRandomRealMatrices(cells, features, 1)[0]
	C := ckks.GenRandomRealMatrices(features, filters, 1)[0]
	W := ckks.GenRandomRealMatrices(filters, classes, 1)[0]
	Y := ckks.GenRandomRealMatrices(1, classes, 1)[0]

	Y.M[0] = 1.0
	Y.M[1] = 0.0

	P := new(ckks.Matrix)
	Ppool := new(ckks.Matrix)
	U := new(ckks.Matrix)
	L1 := new(ckks.Matrix)
	L1Deriv := new(ckks.Matrix)
	E1 := new(ckks.Matrix)
	E0 := new(ckks.Matrix)
	DW := new(ckks.Matrix)
	

	ptL := cellCNN.EncodeLeftForPtMul(L, filters, 1, params)
	ctC := cellCNN.EncryptRightForPtMul(C, cells, params, levelC, sk)
	ptY := cellCNN.EncodeLabelsForBackward(Y, cells,features, filters, classes, params)


	// Returns
	//
	// [[ W transpose row encoded ] [         available         ]]
	//  |    classes * filters    | | Slots - classes * filters | 
	//
	ctW := cellCNN.EncryptRightForNaiveMul(W, params, levelW, sk)


	epoch := 1

	for i := 0; i < epoch; i++{

		// =======================================
		// ========== Plaintext circuit ==========
		// =======================================
		P.MulMat(L, C)

		Ppool.SumColumns(P)
		Ppool.MultConst(Ppool, complex(1.0/float64(cells), 0))
		U.MulMat(Ppool, W)
		
		L1.Func(U, cellCNN.Activation)
		L1Deriv.Func(U, cellCNN.ActivationDeriv)

		E1.Sub(Y, L1)
		E1.Dot(E1, L1Deriv)

		DW.MulMat(Ppool.Transpose(), E1)

		fmt.Println("E1")
		E1.Print()


		fmt.Println("W")
		W.Print()

		E0.MulMat(E1, W.Transpose())

		fmt.Println("E1 x W")
		E0.Print()

		DC := new(ckks.Matrix)
		DC.SumRows(L.Transpose())
		DC.MultConst(DC, complex(1.0/float64(cells), 0))

		fmt.Println("LSum")
		DC.Print()

		DC.MulMat(DC, E0)

		fmt.Println("LSum x (E1 x W)")
		DC.Print()
		

		DW.MultConst(DW, complex(learningRate, 0))



		DC.MultConst(DC, complex(learningRate, 0))

		W.Sub(W, DW)
		C.Sub(C, DC)
		// =======================================

		fmt.Printf("Epoch[%d]\n", i)
		start := time.Now()

		ctTmp := cellCNN.Forward(ptL, ctC, ctW, cells, features, filters, classes, eval, params, sk)

		ctBoot := cellCNN.DummyBoot(ctTmp, L, cells, features, filters, classes, learningRate, params, sk)

		ctDW, ctDC := cellCNN.Backward(ctBoot, ptY, cells, features, filters, classes, params, eval, sk)

		eval.Sub(ctW, ctDW, ctW)
		eval.Sub(ctC, ctDC, ctC)
		
		fmt.Printf("Total : %s\n", time.Since(start))
		fmt.Println()




		
		fmt.Println("DW")
		DW.Print()
		cellCNN.DecryptPrint(0, 16, ctDW, params, sk)

		fmt.Println("DC")
		DC.Print()
		cellCNN.DecryptPrint(0, 16, ctDC, params, sk)
		
	}
}


