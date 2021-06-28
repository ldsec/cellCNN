package centralized

import (
	"fmt"
	"testing"
	"time"

	cl "github.com/ldsec/cellCNN/cellCNNClear/layers"
	"github.com/ldsec/cellCNN/cellcnnPoseidon/utils"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"gonum.org/v1/gonum/mat"
)

func CustomizedParams() ckks.Parameters {
	pl := ckks.ParametersLiteral{
		LogN:     15,
		LogSlots: 14,
		LogQ:     []int{60, 60, 60, 52, 52, 52, 52, 52, 52, 52}, //60*3 + 45*6 = 180 + 270 = 450
		LogP:     []int{61, 61, 61},                             //183
		Scale:    1 << 52,
		Sigma:    rlwe.DefaultSigma,
	}
	params, err := ckks.NewParametersFromLiteral(pl)
	if err != nil {
		panic(err)
	}
	return params
}

func TestWithPlainNetBwBatch(t *testing.T) {
	params := CustomizedParams()

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk)
	encryptor := ckks.NewEncryptorFromSk(params, sk)
	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	ncells := 10
	nmakers := 5
	nfilters := 6
	nclasses := 2
	var sigDegree uint = 3
	sigInterval := 3
	maxM1N2Ratio := 8.0
	momentum := 0.9
	lr := 0.1

	cnnSettings := utils.NewCellCnnSettings(ncells, nmakers, nfilters, nclasses, sigDegree, float64(sigInterval))

	fmt.Printf(
		"settings for cellCNN: ncells: %v | nmakers: %v | nfilters: %v | nclasses: %v\n",
		ncells, nmakers, nfilters, nclasses,
	)
	fmt.Printf(
		"settings for sigmoid least square approximation: degree: %v | interval: %v\n",
		sigDegree, sigInterval,
	)

	// pk := kgen.GenPublicKey(sk)
	cryptoParams := utils.NewCryptoPlaceHolder(params, sk, nil, rlk, encoder, encryptor)

	model := NewCellCNN(cnnSettings, cryptoParams, momentum, lr)
	// cw, dw := model.InitWeights([]*ckks.Ciphertext{ecf1, ecf2}, ecw, append(filter1[:nmakers], filter2[:nmakers]...), weights[:nfilters*nclasses])
	cw, dw := model.InitWeights(nil, nil, nil, nil)
	model.InitEvaluator(cryptoParams, maxM1N2Ratio)

	model.sk = sk

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

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("        TEST BACKWARD OF THE MODEL       ")
	fmt.Println("=========================================")
	fmt.Println()

	batchSize := 2
	iterrations := 10
	isMomentum := false

	var plainOut, dConv, dDense *mat.Dense
	measure := make([]float64, iterrations)
	sumt := 0.0

	for i := 0; i < iterrations; i++ {

		X, matrix, y := utils.GetRandomBatch(nil, batchSize, params, encoder, cnnSettings)
		yMat := utils.ScalarToOneHot(y, nclasses)

		t1 := time.Now()
		// forward & backward
		encOut, _, _ := model.BatchProcessing(X, y, isMomentum)
		if i == 0 {
			plainOut = pNet.ForwardBatch(matrix, cw, dw)
		} else {
			plainOut = pNet.ForwardBatch(matrix, nil, nil)
		}

		// get scaled gradients
		grad := &Gradients{model.conv1d.GetGradient(), model.dense.GetGradient()}

		if model.FisrtMomentum() {
			// if first momentum, btp scaled_g to level 9 and kept as vt
			grad.DummyBootstrapping(encoder, params, sk)
			model.UpdateMomentum(grad)
		} else {
			// else, compute scaled_m at level 8
			grad := model.ComputeScaledGradientWithMomentum(grad, model.cnnSettings, params, model.evaluator, encoder, momentum)
			grad.DummyBootstrapping(encoder, params, sk)
			model.UpdateMomentum(grad)
		}

		model.UpdateWithGradients(grad)

		t2 := time.Since(t1).Seconds()
		measure[i] = t2
		sumt += t2

		errDense := mat.NewDense(batchSize, nclasses, nil)
		errDense.Sub(plainOut, yMat)
		pNet.Backward(errDense, lr, momentum)

		fmt.Printf("ROUND %v ######## Check the forward output #########\n", i)
		utils.DebugWithDense(params, encOut[0], plainOut, decryptor, encoder, 4, []int{0}, true)

		dConv = pNet.conv.GetWeights()
		dDense = pNet.dense.GetWeights()

		fmt.Println("######## Check the backward gradient for filter0 #########")
		utils.DebugCtSliceWithDenseStatistic(params, model.conv1d.GetWeights(), dConv, decryptor, encoder, false, true)
		fmt.Println("######## Check the backward gradient for filter1 #########")
		utils.DebugCtSliceWithDenseStatistic(params, model.conv1d.GetWeights(), dConv, decryptor, encoder, false, true)
		fmt.Println("######## Check the backward gradient for dense #########")
		utils.DebugCtWithDenseStatistic(params, model.dense.GetWeights(), dDense, decryptor, encoder, false, true)

		// runtime.GC()

	}

	fmt.Printf("Time for each iterations: %v, average time for one batch: %v\n", measure, sumt/float64(len(measure)))
}
