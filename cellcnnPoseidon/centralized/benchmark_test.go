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

	t_init_start := time.Now()

	params := CustomizedParams()
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk)
	encryptor := ckks.NewEncryptorFromSk(params, sk)
	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	ncells := 17
	nmakers := 5
	nfilters := 8
	nclasses := 2
	var sigDegree uint = 3
	sigInterval := 5
	maxM1N2Ratio := 8.0
	momentum := 0.9
	lr := 0.1
	batchSize := 3
	iterrations := 3

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

	t_init_end := time.Since(t_init_start).Seconds()

	fmt.Printf(">>>>>> Initialization time (offline): %v\n", t_init_end)

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
	t_fwd_whole := 0.0
	t_bwd_whole := 0.0
	t_root_update := 0.0

	fwd_slice := make([]float64, iterrations)
	bwd_slice := make([]float64, iterrations)
	root_slice := make([]float64, iterrations)

	for i := 0; i < iterrations; i++ {

		X, matrix, y := utils.GetRandomBatch(nil, batchSize, params, encoder, cnnSettings)
		yMat := utils.ScalarToOneHot(y, nclasses)

		// forward & backward
		_, t_fwd_one, t_bwd_one := model.BatchProcessing(X, y, isMomentum)
		t_fwd_whole += t_fwd_one
		t_bwd_whole += t_bwd_one
		fwd_slice[i] = t_fwd_one
		bwd_slice[i] = t_bwd_one

		if i == 0 {
			plainOut = pNet.ForwardBatch(matrix, cw, dw)
		} else {
			plainOut = pNet.ForwardBatch(matrix, nil, nil)
		}

		t_root_start := time.Now()

		// get scaled gradients
		grad := &Gradients{model.conv1d.GetGradient(), model.dense.GetGradient()}

		if model.FisrtMomentum() {
			// if first momentum, btp scaled_g to level 9 and kept as vt
			grad.Bootstrapping(encoder, params, sk)
			model.UpdateMomentum(grad)
		} else {
			// else, compute scaled_m at level 8
			grad = model.ComputeScaledGradientWithMomentum(grad, model.cnnSettings, params, model.evaluator, encoder, momentum)
			grad.Bootstrapping(encoder, params, sk)
			model.UpdateMomentum(grad)
		}

		model.UpdateWithGradients(grad)

		t_root_current := time.Since(t_root_start).Seconds()
		t_root_update += t_root_current
		root_slice[i] = t_root_current

		errDense := mat.NewDense(batchSize, nclasses, nil)
		errDense.Sub(plainOut, yMat)
		pNet.Backward(errDense, lr, momentum)

		dConv = pNet.conv.GetWeights()
		dDense = pNet.dense.GetWeights()

		mse_conv := utils.DebugCtSliceWithDenseStatistic(params, model.conv1d.GetWeights(), dConv, decryptor, encoder, false, verbose)
		mse_dense := utils.DebugCtWithDenseStatistic(params, model.dense.GetWeights(), dDense, decryptor, encoder, false, verbose)

		fmt.Printf(">>> Current Round fwd: %v, bwd: %v, root update: %v, mse_conv: %v, mse_dense: %v\n",
			t_fwd_one, t_bwd_one, t_root_current, mse_conv, mse_dense,
		)

		runtime.GC()

	}

	_, fwd_mse := utils.AVGandStdev(fwd_slice)
	_, bwd_mse := utils.AVGandStdev(bwd_slice)
	_, root_mse := utils.AVGandStdev(root_slice)
	fmt.Println(fwd_slice)

	// fmt.Printf(">>> Initialization time (offline): %v", t_init_end)
	fmt.Printf(">>>>>> Average over %v iteration | fwd: %v, bwd: %v, root update: %v\n",
		iterrations, t_fwd_whole/float64(iterrations), t_bwd_whole/float64(iterrations), t_root_update/float64(iterrations),
	)
	fmt.Printf(">>>>>> MSE over %v iterations | fwd: %v, bwd: %v, root update: %v\n",
		iterrations, fwd_mse, bwd_mse, root_mse,
	)
}

func TestInit(t *testing.T) {
	maxiter := 10
	init_slice := make([]float64, maxiter)
	for i := 0; i < maxiter; i++ {
		t_init_start := time.Now()

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

		t_init_end := time.Since(t_init_start).Seconds()
		init_slice[i] = t_init_end

		fmt.Printf(">>Round %v Initialization time (offline): %v\n", i, t_init_end)

		runtime.GC()
	}
	avg, stdev := utils.AVGandStdev(init_slice)
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

	map_naive_m := make(map[int]float64)
	map_naive_s := make(map[int]float64)
	map_fast_m := make(map[int]float64)
	map_fast_s := make(map[int]float64)

	for k := 1; k <= 5; k++ {
		nclasses = nclasses * 2

		rotIndicesNaive := utils.NewSlice(0, (nfilters-1)*(nclasses-1), nfilters-1)
		rotIndicesFast := params.RotationsForInnerSumLog(nfilters-1, nclasses)
		rotIndices := utils.ClearRotInds(append(rotIndicesNaive, rotIndicesFast...), params.Slots())
		rks := kgen.GenRotationKeysForRotations(rotIndices, false, sk)
		eval := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rks})

		slice_naive := make([]float64, niter)
		slice_fast := make([]float64, niter)
		for i := 0; i < niter; i++ {
			t1 := time.Now()
			_ = utils.MaskAndCollectToLeft(ct, params, encoder, eval, 0, nfilters, nclasses)
			t2 := time.Since(t1).Seconds()

			_ = utils.MaskAndCollectToLeftFast(ct, params, encoder, eval, 0, nfilters, nclasses, true, true)
			t3 := time.Since(t1).Seconds()
			slice_naive[i] = t2
			slice_fast[i] = t3 - t2
		}
		map_naive_m[nclasses], map_naive_s[nclasses] = utils.AVGandStdev(slice_naive)
		map_fast_m[nclasses], map_fast_s[nclasses] = utils.AVGandStdev(slice_fast)
	}

	keys := []int{2, 4, 8, 16, 32}
	prretyPrint(map_naive_m, keys)
	prretyPrint(map_naive_s, keys)
	prretyPrint(map_fast_m, keys)
	prretyPrint(map_fast_s, keys)
	// fmt.Printf("Time: naive: avg: %v, std: %v || fast: avg: %v, std: %v (s)\n", naive_m, naive_s, fast_m, fast_s)

	// fmt.Printf("naive: %v\n", rNaive[:20])
	// fmt.Printf("fast: %v\n", rFast[:20])
	// fmt.Printf("Time: naive: %v, fast: %v (s)\n", t2, t3-t2)
}

func prretyPrint(s map[int]float64, keys []int) {
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
