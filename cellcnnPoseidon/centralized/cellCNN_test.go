package centralized

import (
	"fmt"
	"testing"

	cl "github.com/ldsec/cellCNN/cellCNN_clear/layers"
	"github.com/ldsec/cellCNN/cellCNN_clear/protocols/common"
	"github.com/ldsec/cellCNN/cellcnnPoseidon/layers"
	"github.com/ldsec/cellCNN/cellcnnPoseidon/utils"
	"github.com/ldsec/lattigo/v2/ckks"
	"gonum.org/v1/gonum/mat"
)

func TestOne(t *testing.T) {

	LogN := 14
	LogSlots := 13
	LogModuli := ckks.LogModuli{
		LogQi: []int{55, 40, 40, 40, 40, 40, 40, 40, 40},
		LogPi: []int{30, 30},
	}
	Scale := float64(1 << 40)
	params, err := ckks.NewParametersFromLogModuli(LogN, &LogModuli)
	if err != nil {
		panic(err)
	}
	params.SetScale(Scale)
	params.SetLogSlots(LogSlots)
	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("         INSTANTIATING SCHEME            ")
	fmt.Println("=========================================")
	fmt.Println()

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk)
	encryptor := ckks.NewEncryptorFromSk(params, sk)
	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	ncells := 200
	nmakers := 37
	nfilters := 15
	nclasses := 2
	var sigDegree uint = 3
	sigInterval := 3
	maxM1N2Ratio := 8.0

	cnnSettings := layers.NewCellCnnSettings(ncells, nmakers, nfilters, nclasses, sigDegree, float64(sigInterval))

	fmt.Printf(
		"settings for cellCNN: ncells: %v | nmakers: %v | nfilters: %v | nclasses: %v\n",
		ncells, nmakers, nfilters, nclasses,
	)
	fmt.Printf(
		"settings for sigmoid least square approximation: degree: %v | interval: %v\n",
		sigDegree, sigInterval,
	)

	model := NewCellCNN(cnnSettings, params, rlk, encoder, encryptor)
	model.InitWeights(nil, nil, nil, nil)
	model.InitEvaluator(kgen, sk, encoder, params, maxM1N2Ratio)

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("        TEST FORWARD OF THE MODEL        ")
	fmt.Println("=========================================")
	fmt.Println()

	// create a test input with 2 cells and 4 makers for each cell
	values := make([]complex128, params.Slots())
	for i := range values {
		if i >= ncells*nmakers {
			break
		}
		values[i] = complex(float64(1), 0)
	}

	input := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), values, params.LogSlots())
	output, tt := model.ForwardOne(input, nil, nil, nil, nil)

	fmt.Printf("level of output in forward one: %v\n", output.Level())

	fmt.Printf(
		"Time Consumed: Conv1D: %v | Dense: %v | Sum: %v  (in seconds)\n",
		tt[0], tt[1], tt[2],
	)

	// a := output.El().Value()
	// fmt.Println(a[0].)
	// fmt.Println(output.Element)

	res := make([]complex128, 8)
	res[0] = complex(0.97, 0)
	res[1] = complex(0.97, 0)

	utils.PrintDebug(params, output, res, decryptor, encoder)
}

func TestBatch(t *testing.T) {

	LogN := 14
	LogSlots := 13
	LogModuli := ckks.LogModuli{
		LogQi: []int{55, 40, 40, 40, 40, 40, 40, 40},
		LogPi: []int{45, 45},
	}
	Scale := float64(1 << 40)
	params, err := ckks.NewParametersFromLogModuli(LogN, &LogModuli)
	if err != nil {
		panic(err)
	}
	params.SetScale(Scale)
	params.SetLogSlots(LogSlots)
	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("         INSTANTIATING SCHEME            ")
	fmt.Println("=========================================")
	fmt.Println()

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk)
	encryptor := ckks.NewEncryptorFromSk(params, sk)
	// decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	ncells := 200
	nmakers := 37
	nfilters := 15
	nclasses := 2
	var sigDegree uint = 3
	sigInterval := 7
	maxM1N2Ratio := 8.0

	cnnSettings := layers.NewCellCnnSettings(ncells, nmakers, nfilters, nclasses, sigDegree, float64(sigInterval))

	fmt.Printf(
		"settings for cellCNN: ncells: %v | nmakers: %v | nfilters: %v | nclasses: %v\n",
		ncells, nmakers, nfilters, nclasses,
	)
	fmt.Printf(
		"settings for sigmoid least square approximation: degree: %v | interval: %v\n",
		sigDegree, sigInterval,
	)

	model := NewCellCNN(cnnSettings, params, rlk, encoder, encryptor)
	model.InitWeights(nil, nil, nil, nil)
	model.InitEvaluator(kgen, sk, encoder, params, maxM1N2Ratio)

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("        TEST FORWARD OF THE MODEL        ")
	fmt.Println("=========================================")
	fmt.Println()

	// load the real input dataset
	path := "../../semester_project_claire/data/cellCNN/normalized/"
	trainData := common.LoadTrainDataFrom(path)

	niter := 10
	batchSize := 50

	tt := model.MockTrain(niter, trainData, batchSize)

	fmt.Printf(
		"AVG Time Consumed on each input (ncells * nmakers) During %v iterations with batchSize %v: Conv1D: %v | Dense: %v | Sum: %v\n",
		niter, batchSize, tt[0], tt[1], tt[2],
	)

}

func TestWithPlainNetForward(t *testing.T) {
	LogN := 14
	LogSlots := 13
	LogModuli := ckks.LogModuli{
		LogQi: []int{55, 40, 40, 40, 40, 40, 40, 40},
		LogPi: []int{45, 45},
	}
	// sum of first 3 logQi == Scale +128
	Scale := float64(1 << 40)
	params, err := ckks.NewParametersFromLogModuli(LogN, &LogModuli)
	if err != nil {
		panic(err)
	}
	params.SetScale(Scale)
	params.SetLogSlots(LogSlots)
	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("         INSTANTIATING SCHEME            ")
	fmt.Println("=========================================")
	fmt.Println()

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk)
	encryptor := ckks.NewEncryptorFromSk(params, sk)
	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	ncells := 200
	nmakers := 37
	nfilters := 7
	nclasses := 2
	var sigDegree uint = 3
	sigInterval := 7
	maxM1N2Ratio := 8.0

	cnnSettings := layers.NewCellCnnSettings(ncells, nmakers, nfilters, nclasses, sigDegree, float64(sigInterval))

	fmt.Printf(
		"settings for cellCNN: ncells: %v | nmakers: %v | nfilters: %v | nclasses: %v\n",
		ncells, nmakers, nfilters, nclasses,
	)
	fmt.Printf(
		"settings for sigmoid least square approximation: degree: %v | interval: %v\n",
		sigDegree, sigInterval,
	)

	eNet := NewCellCNN(cnnSettings, params, rlk, encoder, encryptor)
	cw, dw := eNet.InitWeights(nil, nil, nil, nil)
	eNet.InitEvaluator(kgen, sk, encoder, params, maxM1N2Ratio)

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("        TEST FORWARD OF THE MODEL        ")
	fmt.Println("=========================================")
	fmt.Println()

	// load the real input dataset
	path := "../../semester_project_claire/data/cellCNN/normalized/"
	trainData := common.LoadTrainDataFrom(path)

	niter := 10
	batchSize := 5

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

	CompareTwoNetForward(eNet, pNet, cw, dw, trainData, niter, batchSize, decryptor, encoder, params)

}

func TestInnerSum(t *testing.T) {

	LogN := 14
	LogSlots := 13
	LogModuli := ckks.LogModuli{
		LogQi: []int{55, 40, 40, 40, 40, 40, 40, 40},
		LogPi: []int{45, 45},
	}
	Scale := float64(1 << 40)
	params, err := ckks.NewParametersFromLogModuli(LogN, &LogModuli)
	if err != nil {
		panic(err)
	}
	params.SetScale(Scale)
	params.SetLogSlots(LogSlots)
	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("         INSTANTIATING SCHEME            ")
	fmt.Println("=========================================")
	fmt.Println()

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk)
	encryptor := ckks.NewEncryptorFromSk(params, sk)
	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	ncells := 2
	nmakers := 5

	// use predefined weights
	slots := params.Slots()
	filter1 := make([]complex128, slots)
	filter2 := make([]complex128, slots)

	for i, _ := range filter1 {
		if i >= ncells*nmakers {
			break
		}
		filter1[i] = complex(float64(i%nmakers), 0)
		filter2[i] = complex(float64(i%nmakers+1), 0)
	}

	EncodeFilter1 := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), filter1, params.LogSlots())
	EncodeFilter2 := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), filter2, params.LogSlots())

	encFilter1 := encryptor.EncryptNew(EncodeFilter1)
	encFilter2 := encryptor.EncryptNew(EncodeFilter2)

	fmt.Println("weights1: ", filter1[:nmakers*ncells+5])
	fmt.Println("weights2: ", filter2[:nmakers*ncells+5])

	fmt.Println("Conduct innersum on filter1, filter2")
	ind := kgen.GenRotationIndexesForInnerSum(1, ncells*nmakers)
	rks := kgen.GenRotationKeysForRotations(ind, false, sk)

	evaluator := ckks.NewEvaluator(params, ckks.EvaluationKey{Rlk: rlk, Rtks: rks})

	evaluator.InnerSum(encFilter2, 1, nmakers*ncells, encFilter2)
	// evaluator = ckks.NewEvaluator(params, ckks.EvaluationKey{Rlk: rlk, Rtks: rks})
	evaluator.ShallowCopy().InnerSum(encFilter1, 1, nmakers*ncells, encFilter1)

	valuesTest1 := encoder.Decode(decryptor.DecryptNew(encFilter1), params.LogSlots())
	valuesTest2 := encoder.Decode(decryptor.DecryptNew(encFilter2), params.LogSlots())

	fmt.Println("inner filter1: ", valuesTest1[:nmakers*ncells+5])
	fmt.Println("inner filter2: ", valuesTest2[:nmakers*ncells+5])
}

func TestWithPlainNetBwOne(t *testing.T) {
	// LogN := 14
	// LogSlots := 13
	// LogModuli := ckks.LogModuli{
	// 	LogQi: []int{55, 40, 40, 40, 40, 40, 40, 40},
	// 	LogPi: []int{45, 45},
	// }
	// // sum of first 3 logQi == Scale +128
	// Scale := float64(1 << 40)
	// params, err := ckks.NewParametersFromLogModuli(LogN, &LogModuli)
	// if err != nil {
	// 	panic(err)
	// }
	// params.SetScale(Scale)
	// params.SetLogSlots(LogSlots)
	params := ckks.DefaultParams[ckks.PN14QP438]
	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("         INSTANTIATING SCHEME            ")
	fmt.Println("=========================================")
	fmt.Println()

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk)
	encryptor := ckks.NewEncryptorFromSk(params, sk)
	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	ncells := 2
	nmakers := 4
	nfilters := 2
	nclasses := 2
	var sigDegree uint = 3
	sigInterval := 3
	maxM1N2Ratio := 8.0

	cnnSettings := layers.NewCellCnnSettings(ncells, nmakers, nfilters, nclasses, sigDegree, float64(sigInterval))

	fmt.Printf(
		"settings for cellCNN: ncells: %v | nmakers: %v | nfilters: %v | nclasses: %v\n",
		ncells, nmakers, nfilters, nclasses,
	)
	fmt.Printf(
		"settings for sigmoid least square approximation: degree: %v | interval: %v\n",
		sigDegree, sigInterval,
	)

	slots := params.Slots()

	filter1 := make([]complex128, slots)
	for i, _ := range filter1 {
		if i >= ncells*nmakers {
			break
		}
		filter1[i] = complex(float64(i%nmakers)/4, 0)
	}

	filter2 := make([]complex128, slots)
	for i, _ := range filter2 {
		if i >= ncells*nmakers {
			break
		}
		filter2[i] = complex((float64(i%nmakers)+1.0)/4, 0)
	}

	weights := make([]complex128, slots)
	for i, _ := range weights {
		if i >= nfilters*nclasses {
			break
		}
		weights[i] = complex(float64(i%3)/4, 0)
	}

	ef1 := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), filter1, params.LogSlots())
	ef2 := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), filter2, params.LogSlots())
	ew := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), weights, params.LogSlots())
	ecf1 := encryptor.EncryptNew(ef1)
	ecf2 := encryptor.EncryptNew(ef2)
	ecw := encryptor.EncryptNew(ew)

	model := NewCellCNN(cnnSettings, params, rlk, encoder, encryptor)
	cw, dw := model.InitWeights([]*ckks.Ciphertext{ecf1, ecf2}, ecw, append(filter1[:nmakers], filter2[:nmakers]...), weights[:nfilters*nclasses])
	model.InitEvaluator(kgen, sk, encoder, params, maxM1N2Ratio)

	model.sk = sk

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("        TEST BACKWARD OF THE MODEL       ")
	fmt.Println("=========================================")
	fmt.Println()

	// use predefined input row packed with size ncells * nmakers
	plainInput := make([]complex128, slots)
	for i, _ := range plainInput {
		if i >= ncells*nmakers {
			break
		}
		plainInput[i] = complex(float64(i)/5, 0)
	}
	plainInDense := mat.NewDense(ncells, nmakers, utils.SliceCmplxToFloat64(plainInput)[:ncells*nmakers])
	encodeInput := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), plainInput, params.LogSlots())

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

	// forward one get pred
	encOut, _ := model.ForwardOne(encodeInput, nil, nil, nil, nil)
	plainOut := pNet.ForwardBatch([]*mat.Dense{plainInDense}, cw, dw)
	fmt.Println("######## Check the forward output #########")
	utils.DebugWithDense(params, encOut, plainOut, decryptor, encoder, 10, []int{0}, true)

	// validSlotsInds := utils.NewSlice(0, (nclasses-1)*nfilters, nfilters)

	// fmt.Println("######## Check the forward output #########")
	// utils.DebugWithDense(
	// 	params, encOut, plainOut, decryptor, encoder, encInds, plainRows,
	// )

	// utils.DebugWithPlain(
	// 	params, output, outputPlain, decryptor, encoder, validSlotsInds,
	// )

	// labels := make([]float64, 4)
	var label float64 = 1
	err0 := model.ComputeLossOne(encOut, label)

	labelsDense := mat.NewDense(1, nclasses, []float64{0, 1})
	errDense := mat.NewDense(1, nclasses, nil)
	errDense.Sub(plainOut, labelsDense)

	model.BackwardOne(err0)
	pNet.Backward(errDense, 0.6, 0)
	nwfilters := pNet.conv.GetWeights()
	nwdense := pNet.dense.GetWeights()

	model.Step(0.6)

	fmt.Println("######## Check the backward gradient for filter0 #########")
	utils.DebugWithDense(params, model.conv1d.GetWeights()[0], nwfilters, decryptor, encoder, 10, []int{0}, false)
	fmt.Println("######## Check the backward gradient for filter1 #########")
	utils.DebugWithDense(params, model.conv1d.GetWeights()[1], nwfilters, decryptor, encoder, 10, []int{1}, false)
	fmt.Println("######## Check the backward gradient for dense #########")
	utils.DebugWithDense(params, model.dense.GetWeights(), nwdense, decryptor, encoder, 10, []int{0, 1}, false)

	// start at level 9 and scaled gradient end at level 3

	// fmt.Println("######## Check the backward gradient for filter0 #########")
	// utils.DebugWithDense(params, model.conv1d.GetGradient()[0], dConv, decryptor, encoder, 10, []int{0}, false)
	// fmt.Println("######## Check the backward gradient for filter1 #########")
	// utils.DebugWithDense(params, model.conv1d.GetGradient()[1], dConv, decryptor, encoder, 10, []int{1}, false)
	// fmt.Println("######## Check the backward gradient for dense #########")
	// utils.DebugWithDense(params, model.dense.GetGradient(), dDense, decryptor, encoder, 10, []int{0, 1}, false)
}

func TestLargeScale(t *testing.T) {
	params := ckks.DefaultParams[ckks.PN14QP438]
	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("         INSTANTIATING SCHEME            ")
	fmt.Println("=========================================")
	fmt.Println()

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk)
	encryptor := ckks.NewEncryptorFromSk(params, sk)
	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	ncells := 200
	nmakers := 37
	nfilters := 7
	nclasses := 2
	var sigDegree uint = 3
	sigInterval := 7
	maxM1N2Ratio := 8.0

	cnnSettings := layers.NewCellCnnSettings(ncells, nmakers, nfilters, nclasses, sigDegree, float64(sigInterval))

	fmt.Printf(
		"settings for cellCNN: ncells: %v | nmakers: %v | nfilters: %v | nclasses: %v\n",
		ncells, nmakers, nfilters, nclasses,
	)
	fmt.Printf(
		"settings for sigmoid least square approximation: degree: %v | interval: %v\n",
		sigDegree, sigInterval,
	)

	eNet := NewCellCNN(cnnSettings, params, rlk, encoder, encryptor)
	cw, dw := eNet.InitWeights(nil, nil, nil, nil)
	eNet.InitEvaluator(kgen, sk, encoder, params, maxM1N2Ratio)

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("        TEST FORWARD OF THE MODEL        ")
	fmt.Println("=========================================")
	fmt.Println()

	// load the real input dataset
	path := "../../semester_project_claire/data/cellCNN/normalized/"
	trainData := common.LoadTrainDataFrom(path)

	niter := 10
	batchSize := 5
	lr := 0.6

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

	CompareTwoNetBackward(eNet, pNet, cw, dw, trainData, niter, batchSize, lr, decryptor, encoder, params)
}

func TestTimeForwardBackward(t *testing.T) {
	params := ckks.DefaultParams[ckks.PN14QP438]
	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("         INSTANTIATING SCHEME            ")
	fmt.Println("=========================================")
	fmt.Println()

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk)
	encryptor := ckks.NewEncryptorFromSk(params, sk)
	// decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	ncells := 20
	nmakers := 8
	nfilters := 7
	nclasses := 2
	var sigDegree uint = 3
	sigInterval := 7
	maxM1N2Ratio := 8.

	cnnSettings := layers.NewCellCnnSettings(ncells, nmakers, nfilters, nclasses, sigDegree, float64(sigInterval))

	fmt.Printf(
		"settings for cellCNN: ncells: %v | nmakers: %v | nfilters: %v | nclasses: %v\n",
		ncells, nmakers, nfilters, nclasses,
	)
	fmt.Printf(
		"settings for sigmoid least square approximation: degree: %v | interval: %v\n",
		sigDegree, sigInterval,
	)

	eNet := NewCellCNN(cnnSettings, params, rlk, encoder, encryptor)
	eNet.InitWeights(nil, nil, nil, nil)
	eNet.InitEvaluator(kgen, sk, encoder, params, maxM1N2Ratio)
	eNet.sk = sk

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("        TEST FORWARD OF THE MODEL        ")
	fmt.Println("=========================================")
	fmt.Println()

	// load the real input dataset
	// path := "../../semester_project_claire/data/cellCNN/normalized/"
	// trainData := common.LoadTrainDataFrom(path)

	niter := 10
	batchSize := 5
	// lr := 0.6

	ForwardAndBackwardNiter(eNet, niter, batchSize)
}
