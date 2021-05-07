package centralized

import (
	"fmt"
	"testing"

	"github.com/ldsec/cellCNN/cellcnnPoseidon/layers"
	"github.com/ldsec/cellCNN/cellcnnPoseidon/utils"
	"github.com/ldsec/lattigo/v2/ckks"
)

func TestDenseBackwardOne(t *testing.T) {

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
	nfilters := 3
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
	fmt.Println("        TEST BACKWARD OF THE MODEL       ")
	fmt.Println("=========================================")
	fmt.Println()

	// require:
	// 	input *ckksciphertext with conv1d activation on left most k slots
	// 	weights *ckksciphertext with column packing

	// use predefined weights, column packed 3*2
	slots := params.Slots()
	plainWeights := make([]complex128, slots)
	for i, _ := range plainWeights {
		if i >= nfilters*nclasses {
			break
		}
		plainWeights[i] = complex(float64(i%4), 0)
	}
	encodeWeights := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), plainWeights, params.LogSlots())
	encryptWeights := encryptor.EncryptNew(encodeWeights)

	// use predefined input, one row 1*3
	plainInput := make([]complex128, slots)
	for i, _ := range plainInput {
		if i >= nfilters {
			break
		}
		plainInput[i] = complex(float64(i), 0)
	}

	encodeInput := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), plainInput, params.LogSlots())
	encryptInput := encryptor.EncryptNew(encodeInput)

	fmt.Printf("Input: %v\nWeights: %v\n", plainInput[:10], plainWeights[:10])

	// retrive the dense layer
	ds := model.dense
	ds.WithWeights(encryptWeights)

	// cipher / plaintext forward compare
	outputEnc := ds.Forward(encryptInput, encryptWeights, model.cnnSettings, model.evaluator, model.encoder, model.params, model.GenerateMaskMap())

	outputPlain, plainU := ds.PlainForwardCircuit(plainWeights, plainInput, model.cnnSettings)
	validSlotsInds := utils.NewSlice(0, (nclasses-1)*nfilters, nfilters)
	// fmt.Printf("valid: %v\n", validSlotsInds)

	utils.DebugWithPlain(params, outputEnc, outputPlain, decryptor, encoder, validSlotsInds)

	// #####################################################################################

	// cipher / plaintext backward compare
	// 1. prepare the err: 2(pred-y)
	// use predefined input, one row 1*3
	Err0 := make([]complex128, slots)
	Err0[0] = complex(0.5, 0)
	Err0[3] = complex(0.2, 0)

	encodeErr0 := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), Err0, params.LogSlots())
	encryptErr0 := encryptor.EncryptNew(encodeErr0)

	fmt.Printf("Dense input err: Level: %d (logQ = %d)\n", encryptErr0.Level(), params.LogQLvl(encryptErr0.Level()))

	// 2. compute the d approximate sigmoid
	_, mErr_plain_circuit := ds.PlainBackwardCircuit(plainWeights, plainInput, plainU, Err0, model.cnnSettings)
	mErr_encrypt_circuit := ds.Backward(encryptErr0, model.cnnSettings, params, model.evaluator, encoder, nil)

	utils.DebugWithPlain(params, mErr_encrypt_circuit, mErr_plain_circuit, decryptor, encoder, validSlotsInds)
}

func TestConv1dBackwardOne(t *testing.T) {

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("         INSTANTIATING SCHEME            ")
	fmt.Println("=========================================")
	fmt.Println()

	params := ckks.DefaultParams[ckks.PN14QP438]
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk)
	encryptor := ckks.NewEncryptorFromSk(params, sk)
	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	ncells := 2
	nmakers := 4
	nfilters := 3
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
	fmt.Println("        TEST BACKWARD OF THE MODEL       ")
	fmt.Println("=========================================")
	fmt.Println()

	// require:
	// input plaintext: with valid slots at:
	// err ciphertext:  with valid slots at: nclasses * (0~k-1)

	// use predefined weights, column packed
	slots := params.Slots()

	// use predefined input row packed with size ncells * nmakers
	plainInput := make([]complex128, slots)
	for i, _ := range plainInput {
		if i >= ncells*nmakers {
			break
		}
		plainInput[i] = complex(float64(i), 0)
	}

	encodeInput := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), plainInput, params.LogSlots())
	// encryptInput := encryptor.EncryptNew(encodeInput)

	fmt.Printf("Input: %v\n", plainInput[:10])

	// retrive the dense layer
	cv := model.conv1d
	cv.WithLastInput(encodeInput)
	cv.WithEncoder(encoder)
	// ds.WithWeights(encryptWeights)

	// cipher / plaintext backward compare
	// 1. use predefined err, one row 1*3
	Err0 := make([]complex128, slots)
	Err0[0] = complex(0.5, 0)
	Err0[2] = complex(0.2, 0)
	Err0[4] = complex(0.3, 0)

	encodeErr0 := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), Err0, params.LogSlots())
	encryptErr0 := encryptor.EncryptNew(encodeErr0)

	fmt.Printf("Conv1d input err: Level: %d (logQ = %d)\n", encryptErr0.Level(), params.LogQLvl(encryptErr0.Level()))

	// 2. compute the d approximate sigmoid
	mErr_plain_circuit := cv.PlainBackwardCircuit(plainInput, Err0, model.cnnSettings)
	mErr_encrypt_circuit := cv.Backward(encryptErr0, model.cnnSettings, params, model.evaluator, encoder)

	validSlotsInds := utils.NewSlice(0, ncells*nmakers-1, 1)

	for i := 0; i < model.cnnSettings.Nfilters; i++ {
		fmt.Printf("\n\n==> Gradients of Filter ID: %d", i)
		utils.DebugWithPlain(
			params, mErr_encrypt_circuit[i], mErr_plain_circuit[i], decryptor, encoder, validSlotsInds,
		)
	}
}

func TestConv1dForwardOne(t *testing.T) {

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("         INSTANTIATING SCHEME            ")
	fmt.Println("=========================================")
	fmt.Println()

	params := ckks.DefaultParams[ckks.PN14QP438]
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
	fmt.Println("        TEST FORWARD OF THE CONV1D       ")
	fmt.Println("=========================================")
	fmt.Println()

	// use predefined weights, column packed
	slots := params.Slots()

	// use predefined input row packed with size ncells * nmakers
	plainInput := make([]complex128, slots)
	for i, _ := range plainInput {
		if i >= ncells*nmakers {
			break
		}
		plainInput[i] = complex(float64(i), 0)
	}

	filter1 := make([]complex128, slots)
	for i, _ := range filter1 {
		if i >= ncells*nmakers {
			break
		}
		filter1[i] = complex(float64(i%nmakers), 0)
	}

	filter2 := make([]complex128, slots)
	for i, _ := range filter2 {
		if i >= ncells*nmakers {
			break
		}
		filter2[i] = complex(float64(i%nmakers)+1.0, 0)
	}

	fmt.Println("FILTER1: ", filter1[:10])
	fmt.Println("FILTER2: ", filter2[:10])
	fmt.Println("INPUT: ", plainInput[:10])

	encodeInput := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), plainInput, params.LogSlots())
	ef1 := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), filter1, params.LogSlots())
	ef2 := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), filter2, params.LogSlots())
	ecf1 := encryptor.EncryptNew(ef1)
	ecf2 := encryptor.EncryptNew(ef2)

	// retrive the dense layer
	cv := model.conv1d
	cv.WithLastInput(encodeInput)
	cv.WithEncoder(encoder)
	cv.WithWeights([]*ckks.Ciphertext{ecf1, ecf2})
	// ds.WithWeights(encryptWeights)

	// 2. compute the d approximate sigmoid
	out_plain_circuit := cv.PlainForwardCircuit(plainInput, [][]complex128{filter1, filter2}, model.cnnSettings)
	out_encrypt_circuit := model.ForwardConv(encodeInput, nil, nil, nil, nil)

	validSlotsInds := utils.NewSlice(0, ncells*nmakers-1, 1)

	utils.DebugWithPlain(
		params, out_encrypt_circuit, out_plain_circuit, decryptor, encoder, validSlotsInds,
	)

}

func TestModelForwardBackwardOne(t *testing.T) {

	// btpParams := ckks.DefaultBootstrapParams[0]
	// params, _ := btpParams.Params()
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
	fmt.Println("Done")

	ncells := 2
	nmakers := 4
	nfilters := 2
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
	model.InitWeights([]*ckks.Ciphertext{ecf1, ecf2}, ecw, nil, nil)
	model.InitEvaluator(kgen, sk, encoder, params, maxM1N2Ratio)
	model.conv1d.WithEncoder(encoder)

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
	encodeInput := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), plainInput, params.LogSlots())

	// init plaintext model
	model.pcir = NewPlainCircuit([][]complex128{filter1, filter2}, weights, plainInput)

	// forward one get pred
	output, _ := model.ForwardOne(encodeInput, nil, nil, nil, nil)
	outputPlain := model.PlaintextCircuitForwardOne()

	validSlotsInds := utils.NewSlice(0, (nclasses-1)*nfilters, nfilters)

	fmt.Println("######## Check the forward output #########")
	utils.DebugWithPlain(
		params, output, outputPlain, decryptor, encoder, validSlotsInds,
	)

	labels := make([]float64, 4)
	labels[0] = 0.6
	labels[2] = 0.4
	err0 := model.ComputeLossOne(output, 1)
	model.BackwardOne(err0)

	perr := utils.PlaintextLoss(outputPlain, labels)
	df, dw := model.PlaintextCircuitBackwardOne(perr)

	fmt.Println("######## Check the gradient of conv1d #########")

	utils.DebugWithPlain(
		params, model.conv1d.GetGradient()[0], df[0], decryptor, encoder, validSlotsInds,
	)

	fmt.Println("######## Check the gradient of dense #########")

	utils.DebugWithPlain(
		params, model.dense.GetGradient(), dw, decryptor, encoder, validSlotsInds,
	)

	// model.Step(0.1)
}
