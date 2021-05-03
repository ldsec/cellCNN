package test

// import (
// 	"fmt"
// 	"math"
// 	"testing"

// 	"github.com/ldsec/lattigo/v2/ckks"
// 	"github.com/ldsec/cellCNN/cellcnnPoseidon/layers"
// )

// func PrintDebug(params *ckks.Parameters, ciphertext *ckks.Ciphertext, valuesWant []complex128, decryptor ckks.Decryptor, encoder ckks.Encoder) (valuesTest []complex128) {

// 	valuesTest = encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())

// 	fmt.Println()
// 	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
// 	fmt.Printf("Scale: 2^%f\n", math.Log2(ciphertext.Scale()))
// 	fmt.Printf("Activation Test: %v ...\n", valuesTest[0:8])
// 	fmt.Printf("Activation Want: %v ...\n", valuesWant[0:8])
// 	fmt.Println()

// 	// precStats := ckks.GetPrecisionStats(params, nil, nil, valuesWant, valuesTest)

// 	// fmt.Println(precStats.String())

// 	return
// }

// // TestForward test forward in conv1d and avg pool
// func TestCorrectness(t *testing.T) {
// 	LogN := 14
// 	LogSlots := 13

// 	LogModuli := ckks.LogModuli{
// 		LogQi: []int{55, 40, 40, 40, 40, 40, 40, 40},
// 		LogPi: []int{45, 45},
// 	}

// 	Scale := float64(1 << 40)

// 	params, err := ckks.NewParametersFromLogModuli(LogN, &LogModuli)
// 	if err != nil {
// 		panic(err)
// 	}
// 	params.SetScale(Scale)
// 	params.SetLogSlots(LogSlots)

// 	fmt.Println()
// 	fmt.Println("=========================================")
// 	fmt.Println("         INSTANTIATING SCHEME            ")
// 	fmt.Println("=========================================")
// 	fmt.Println()

// 	kgen := ckks.NewKeyGenerator(params)
// 	sk := kgen.GenSecretKey()
// 	rlk := kgen.GenRelinearizationKey(sk)
// 	encryptor := ckks.NewEncryptorFromSk(params, sk)
// 	decryptor := ckks.NewDecryptor(params, sk)
// 	encoder := ckks.NewEncoder(params)

// 	slots := params.Slots()

// 	ncells := 2
// 	nmakers := 4

// 	inds := kgen.GenRotationIndexesForInnerSum(1, 4)
// 	rks := kgen.GenRotationKeysForRotations(inds, false, sk)

// 	evaluator := ckks.NewEvaluator(params, ckks.EvaluationKey{Rlk: rlk, Rtks: rks})

// 	// create a test input with 2 cells and 4 makers for each cell
// 	values := make([]complex128, slots)
// 	for i := range values {
// 		if i >= ncells*nmakers {
// 			break
// 		}
// 		values[i] = complex(float64(i), 0)
// 	}

// 	plainX := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), values, params.LogSlots())

// 	filter1 := make([]complex128, slots)
// 	// filter2 := make([]complex128, slots)

// 	for i := range filter1 {
// 		if i >= ncells*nmakers {
// 			break
// 		}
// 		filter1[i] = complex(float64(i%nmakers), 0)
// 		// filter2[i] = complex(float64(i%nmakers+1), 0)
// 	}

// 	EncodeFilter1 := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), filter1, params.LogSlots())
// 	// EncodeFilter2 := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), filter2, params.LogSlots())

// 	encFilter1 := encryptor.EncryptNew(EncodeFilter1)
// 	// encFilter2 := encryptor.EncryptNew(EncodeFilter2)

// 	out1 := evaluator.MulRelinNew(encFilter1, plainX)
// 	evaluator.InnerSum(out1, 1, 4, out1)

// 	PrintDebug(params, out1, values, decryptor, encoder)
// }

// // TestForward test forward in conv1d and avg pool
// func TestForwad(t *testing.T) {
// 	LogN := 14
// 	LogSlots := 13

// 	LogModuli := ckks.LogModuli{
// 		LogQi: []int{55, 40, 40, 40, 40, 40, 40, 40},
// 		LogPi: []int{45, 45},
// 	}

// 	Scale := float64(1 << 40)

// 	params, err := ckks.NewParametersFromLogModuli(LogN, &LogModuli)
// 	if err != nil {
// 		panic(err)
// 	}
// 	params.SetScale(Scale)
// 	params.SetLogSlots(LogSlots)

// 	fmt.Println()
// 	fmt.Println("=========================================")
// 	fmt.Println("         INSTANTIATING SCHEME            ")
// 	fmt.Println("=========================================")
// 	fmt.Println()

// 	kgen := ckks.NewKeyGenerator(params)
// 	sk := kgen.GenSecretKey()
// 	rlk := kgen.GenRelinearizationKey(sk)
// 	encryptor := ckks.NewEncryptorFromSk(params, sk)
// 	decryptor := ckks.NewDecryptor(params, sk)
// 	encoder := ckks.NewEncoder(params)

// 	slots := params.Slots()

// 	ncells := 2
// 	nmakers := 4
// 	nfilters := 2
// 	nclasses := 2
// 	var sigDegree uint = 3
// 	sigInterval := 7

// 	indsConv := kgen.GenRotationIndexesForInnerSum(1, nmakers*ncells)
// 	indsDense := kgen.GenRotationIndexesForInnerSum(1, nfilters)
// 	rotConv := make([]int, 0)
// 	for i := 1; i < nfilters; i++ {
// 		rotConv = append(rotConv, -i)
// 	}
// 	indsDense = append(indsDense, rotConv...)
// 	for i := 1; i < nclasses; i++ {
// 		indsDense = append(indsDense, -i*nfilters)
// 	}
// 	step := nfilters - 1
// 	rotSlice := layers.NewSlice(0, (nclasses-1)*step, step)
// 	indsDense = append(indsDense, rotSlice...)

// 	rks := kgen.GenRotationKeysForRotations(append(indsDense, indsConv...), false, sk)

// 	evaluator := ckks.NewEvaluator(params, ckks.EvaluationKey{Rlk: rlk, Rtks: rks})

// 	// create a test input with 2 cells and 4 makers for each cell
// 	values := make([]complex128, slots)
// 	for i := range values {
// 		if i >= ncells*nmakers {
// 			break
// 		}
// 		values[i] = complex(float64(i), 0)
// 	}

// 	plainX := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), values, params.LogSlots())

// 	filter1 := make([]complex128, slots)
// 	filter2 := make([]complex128, slots)

// 	for i := range filter1 {
// 		if i >= ncells*nmakers {
// 			break
// 		}
// 		filter1[i] = complex(float64(i%nmakers), 0)
// 		filter2[i] = complex(float64(i%nmakers+1), 0)
// 	}

// 	denseWeights := make([]complex128, slots)

// 	for i := range denseWeights {
// 		if i >= nfilters*nclasses {
// 			break
// 		}
// 		if i%2 == 0 {
// 			denseWeights[i] = complex(-0.25, 0)
// 		} else {
// 			denseWeights[i] = complex(0.25, 0)
// 		}
// 	}

// 	EncodeFilter1 := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), filter1, params.LogSlots())
// 	EncodeFilter2 := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), filter2, params.LogSlots())
// 	EncodeWeight := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), denseWeights, params.LogSlots())

// 	encFilter1 := encryptor.EncryptNew(EncodeFilter1)
// 	encFilter2 := encryptor.EncryptNew(EncodeFilter2)
// 	encWeight := encryptor.EncryptNew(EncodeWeight)

// 	fmt.Printf(
// 		"fresh weights scale: %v, %v, %v\n",
// 		math.Log2(encFilter1.Scale()), math.Log2(encFilter2.Scale()),
// 		math.Log2(encWeight.Scale()),
// 	)

// 	// mask for pooling
// 	LeftMostMask := make([]complex128, params.Slots())
// 	LeftMostMask[0] = complex(float64(1), 0)
// 	poolMask := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), LeftMostMask, params.LogSlots())

// 	// Dense: maskMap
// 	maskMap := make(map[int]*ckks.Plaintext)
// 	for i := 0; i < nclasses; i++ {
// 		maskMap[i*(nfilters-1)] = func() *ckks.Plaintext {
// 			tmpMask := make([]complex128, params.Slots())
// 			fmt.Println("making maskMap: ", i*(nfilters-1))
// 			tmpMask[i] = complex(float64(1), 0)
// 			return encoder.EncodeNTTAtLvlNew(params.MaxLevel(), tmpMask, params.LogSlots())
// 		}()
// 	}

// 	conv1d := layers.NewConv1D(ncells, nmakers, []*ckks.Ciphertext{encFilter1, encFilter2})
// 	dense := layers.NewDense(nclasses, encWeight, nfilters, nmakers, sigDegree, float64(sigInterval))

// 	activations, ev := conv1d.Forward(plainX, nil, evaluator, poolMask, params)

// 	fmt.Printf(
// 		"conv activation scale: %v\n",
// 		math.Log2(activations.Scale()),
// 	)

// 	output := dense.Forward(activations, nil, ev, params, maskMap)

// 	PrintDebug(params, output, values, decryptor, encoder)
// }
