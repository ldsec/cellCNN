package centralized

import (
	"fmt"
	"runtime"
	"testing"
	"time"

	cl "github.com/ldsec/cellCNN/cellCNN_clear/layers"
	"github.com/ldsec/cellCNN/cellcnnPoseidon/utils"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"gonum.org/v1/gonum/mat"
)

func TestLocalTime(t *testing.T) {

	tInitStart := time.Now()

	params := CustomizedParams()
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk)
	encryptor := ckks.NewEncryptorFromSk(params, sk)
	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	ncells := 200
	nmakers := 38
	nfilters := 4
	nclasses := 2
	var sigDegree uint = 3
	sigInterval := 5
	maxM1N2Ratio := 8.0
	momentum := 0.9
	lr := 0.1
	batchSize := 10
	iterrations := 50

	cnnSettings := utils.NewCellCnnSettings(ncells, nmakers, nfilters, nclasses, sigDegree, float64(sigInterval))

	fmt.Printf(
		"> Settings for cellCNN: ncells: %v | nmakers: %v | nfilters: %v | nclasses: %v\n",
		ncells, nmakers, nfilters, nclasses,
	)
	fmt.Printf(
		"> Settings for training: degree: %v | interval: %v | max iter: %v | batchsize: %v | lr: %v, momentum: %v\n",
		sigDegree, sigInterval, iterrations, batchSize, lr, momentum,
	)

	cryptoParams := utils.NewCryptoPlaceHolder(params, sk, nil, rlk, encoder, encryptor)
	model := NewCellCNN(cnnSettings, cryptoParams, momentum, lr)
	cw, dw := model.InitWeights(nil, nil, nil, nil)
	model.InitEvaluator(cryptoParams, maxM1N2Ratio)
	model.sk = sk

	tInitEnd := time.Since(tInitStart).Seconds()

	fmt.Printf(">>>>>> Initialization time (offline): %v\n", tInitEnd)

	// init plaintext net
	pconv := &cl.Conv1D{Nfilters: nfilters}
	ppool := &cl.Pool{}
	pdense := &cl.Dense_n{Nclasses: nclasses, ApproxInterval: float64(sigInterval)}

	pNet := &PlainNet{
		ncells:   ncells,
		nmakers:  nmakers,
		nfilters: nfilters,
		nclasses: nclasses,
		conv:     pconv,
		pool:     ppool,
		dense:    pdense,
	}

	var plainOut, dConv, dDense *mat.Dense

	// supress the local momentum
	isMomentum := false
	verbose := true
	tFwdWhole := 0.0
	tBwdWhole := 0.0
	tRootUpdate := 0.0

	fwdSlice := make([]float64, iterrations)
	bwdSlice := make([]float64, iterrations)
	rootSlice := make([]float64, iterrations)
	errConvRecords := make(map[int]float64)
	errDenseRecords := make(map[int]float64)

	for i := 0; i < iterrations; i++ {

		X, matrix, y := utils.GetRandomBatch(nil, batchSize, params, encoder, cnnSettings)
		yMat := utils.ScalarToOneHot(y, nclasses)

		// forward & backward
		_, tFwdOne, tBwdOne := model.BatchProcessing(X, y, isMomentum)
		tFwdWhole += tFwdOne
		tBwdWhole += tBwdOne
		fwdSlice[i] = tFwdOne
		bwdSlice[i] = tBwdOne

		if i == 0 {
			plainOut = pNet.ForwardBatch(matrix, cw, dw)
		} else {
			plainOut = pNet.ForwardBatch(matrix, nil, nil)
		}

		tRootStart := time.Now()

		// get scaled gradients
		grad := &Gradients{model.conv1d.GetGradient(), model.dense.GetGradient()}

		if model.FisrtMomentum() {
			// if first momentum, btp scaledG to level 9 and kept as vt
			grad.Bootstrapping(encoder, params, sk)
			model.UpdateMomentum(grad)
		} else {
			// else, compute scaledM at level 8
			grad = model.ComputeScaledGradientWithMomentum(grad, model.cnnSettings, params, model.evaluator, encoder, momentum)
			grad.Bootstrapping(encoder, params, sk)
			model.UpdateMomentum(grad)
		}

		model.UpdateWithGradients(grad)

		tRootCurrent := time.Since(tRootStart).Seconds()
		tRootUpdate += tRootCurrent
		rootSlice[i] = tRootCurrent

		errDense := mat.NewDense(batchSize, nclasses, nil)
		errDense.Sub(plainOut, yMat)
		pNet.Backward(errDense, lr, momentum)

		dConv = pNet.conv.GetWeights()
		dDense = pNet.dense.GetWeights()

		mseConv := utils.DebugCtSliceWithDenseStatistic(params, model.conv1d.GetWeights(), dConv, decryptor, encoder, false, verbose)
		mseDense := utils.DebugCtWithDenseStatistic(params, model.dense.GetWeights(), dDense, decryptor, encoder, false, verbose)

		fmt.Printf(">>> Current Round fwd: %v, bwd: %v, root update: %v, mseConv: %v, mseDense: %v\n",
			tFwdOne, tBwdOne, tRootCurrent, mseConv, mseDense,
		)

		errConvRecords[i] = mseConv
		errDenseRecords[i] = mseDense

		runtime.GC()

	}

	_, fwdMse := utils.AVGandStdev(fwdSlice)
	_, bwdMse := utils.AVGandStdev(bwdSlice)
	_, rootMse := utils.AVGandStdev(rootSlice)

	// fmt.Printf(">>> Initialization time (offline): %v", tInitEnd)
	fmt.Printf(">>>>>> Average over %v iteration | fwd: %v, bwd: %v, root update: %v\n",
		iterrations, tFwdWhole/float64(iterrations), tBwdWhole/float64(iterrations), tRootUpdate/float64(iterrations),
	)
	fmt.Printf(">>>>>> MSE over %v iterations | fwd: %v, bwd: %v, root update: %v\n",
		iterrations, fwdMse, bwdMse, rootMse,
	)

	fmt.Println("conv err")
	prettyPrint(errConvRecords, utils.NewSlice(0, iterrations-1, 1))
	fmt.Println("dense err")
	prettyPrint(errDenseRecords, utils.NewSlice(0, iterrations-1, 1))

}

func TestInit(t *testing.T) {
	maxiter := 10
	initSlice := make([]float64, maxiter)
	for i := 0; i < maxiter; i++ {
		tInitStart := time.Now()

		params := CustomizedParams()
		kgen := ckks.NewKeyGenerator(params)
		sk := kgen.GenSecretKey()
		rlk := kgen.GenRelinearizationKey(sk)
		encryptor := ckks.NewEncryptorFromSk(params, sk)
		// decryptor := ckks.NewDecryptor(params, sk)
		encoder := ckks.NewEncoder(params)

		ncells := 200
		nmakers := 38
		nfilters := 8
		nclasses := 2
		var sigDegree uint = 3
		sigInterval := 5
		maxM1N2Ratio := 8.0
		momentum := 0.9
		lr := 0.1
		// batchSize := 1
		// iterrations := 10

		cnnSettings := utils.NewCellCnnSettings(ncells, nmakers, nfilters, nclasses, sigDegree, float64(sigInterval))

		// fmt.Printf(
		// 	"> Settings for cellCNN: ncells: %v | nmakers: %v | nfilters: %v | nclasses: %v\n",
		// 	ncells, nmakers, nfilters, nclasses,
		// )
		// fmt.Printf(
		// 	"> Settings for training: degree: %v | interval: %v | max iter: %v | batchsize: %v | lr: %v, momentum: %v\n",
		// 	sigDegree, sigInterval, iterrations, batchSize, lr, momentum,
		// )

		cryptoParams := utils.NewCryptoPlaceHolder(params, sk, nil, rlk, encoder, encryptor)
		model := NewCellCNN(cnnSettings, cryptoParams, momentum, lr)
		model.InitWeights(nil, nil, nil, nil)
		model.InitEvaluator(cryptoParams, maxM1N2Ratio)
		model.sk = sk

		tInitEnd := time.Since(tInitStart).Seconds()
		initSlice[i] = tInitEnd

		fmt.Printf(">>Round %v Initialization time (offline): %v\n", i, tInitEnd)

		runtime.GC()
	}
	avg, stdev := utils.AVGandStdev(initSlice)
	fmt.Printf(">> Initialization time (offline) | avg: %v, stdev: %v\n", avg, stdev)
}

func TestOptimizedCollect(t *testing.T) {
	params := CustomizedParams()
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk)
	encryptor := ckks.NewEncryptorFromSk(params, sk)
	// decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	nfilters := 33

	clear := make([]complex128, params.Slots())
	for i := 0; i < 10; i++ {
		clear[i*nfilters] = 1
	}
	plain := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), clear, params.LogSlots())
	ct := encryptor.EncryptNew(plain)

	nclasses := 1
	niter := 10

	mapNaiveM := make(map[int]float64)
	mapNaiveS := make(map[int]float64)
	mapFastM := make(map[int]float64)
	mapFastS := make(map[int]float64)

	for k := 1; k <= 5; k++ {
		nclasses = nclasses * 2

		rotIndicesNaive := utils.NewSlice(0, (nfilters-1)*(nclasses-1), nfilters-1)
		rotIndicesFast := params.RotationsForInnerSumLog(nfilters-1, nclasses)
		rotIndices := utils.ClearRotInds(append(rotIndicesNaive, rotIndicesFast...), params.Slots())
		rks := kgen.GenRotationKeysForRotations(rotIndices, false, sk)
		eval := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rks})

		sliceNaive := make([]float64, niter)
		sliceFast := make([]float64, niter)
		for i := 0; i < niter; i++ {
			t1 := time.Now()
			_ = utils.MaskAndCollectToLeft(ct, params, encoder, eval, 0, nfilters, nclasses)
			t2 := time.Since(t1).Seconds()

			_ = utils.MaskAndCollectToLeftFast(ct, params, encoder, eval, 0, nfilters, nclasses, true, true)
			t3 := time.Since(t1).Seconds()
			sliceNaive[i] = t2
			sliceFast[i] = t3 - t2
		}
		mapNaiveM[nclasses], mapNaiveS[nclasses] = utils.AVGandStdev(sliceNaive)
		mapFastM[nclasses], mapFastS[nclasses] = utils.AVGandStdev(sliceFast)
	}

	keys := []int{2, 4, 8, 16, 32}
	prettyPrint(mapNaiveM, keys)
	prettyPrint(mapNaiveS, keys)
	prettyPrint(mapFastM, keys)
	prettyPrint(mapFastS, keys)
	// fmt.Printf("Time: naive: avg: %v, std: %v || fast: avg: %v, std: %v (s)\n", naiveM, naiveS, fastM, fastS)

	// fmt.Printf("naive: %v\n", rNaive[:20])
	// fmt.Printf("fast: %v\n", rFast[:20])
	// fmt.Printf("Time: naive: %v, fast: %v (s)\n", t2, t3-t2)
}

func prettyPrint(s map[int]float64, keys []int) {
	res := "["
	for i, key := range keys {
		if i != len(keys)-1 {
			res += fmt.Sprint(s[key]) + ", "
		} else {
			res += fmt.Sprint(s[key])
		}
	}
	fmt.Println(res + "]")
}
