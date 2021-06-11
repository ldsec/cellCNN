package test

// import (
// 	"fmt"
// 	"math"
// 	"testing"

// 	"github.com/ldsec/cellCNN/cellcnnPoseidon/utils"
// 	"github.com/ldsec/lattigo/v2/ckks"
// )

// func PrintDebugMatrix(
// 	params ckks.Parameters, ciphertext *ckks.Ciphertext,
// 	valuesWant []complex128, decryptor ckks.Decryptor,
// 	encoder ckks.Encoder, r, c int, inRowPacked bool, ouRowPacked bool,
// ) (valuesTest []complex128) {

// 	valuesTest = encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())

// 	mt, vc := DebugPlaintextMatrix(valuesWant, r, c, inRowPacked, ouRowPacked)

// 	fmt.Println()
// 	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
// 	fmt.Printf("Scale: 2^%f\n", math.Log2(ciphertext.Scale()))
// 	fmt.Printf("Activation Test: %v ...\n", valuesTest[0:8])
// 	fmt.Printf("Activation Want: %v ...\n", vc)
// 	fmt.Println("transpose matrix: ")
// 	for i, v := range mt {
// 		fmt.Printf("r<%d>: %v\n", i, v)
// 	}
// 	fmt.Println()

// 	// precStats := ckks.GetPrecisionStats(params, nil, nil, valuesWant, valuesTest)

// 	// fmt.Println(precStats.String())

// 	return
// }

// func GetRotationInds(diag map[int][]complex128) []int {
// 	// get the rotation records
// 	keys := make([]int, len(diag))
// 	i := 0
// 	for k := range diag {
// 		keys[i] = k
// 		i++
// 	}
// 	return keys
// }

// func DebugPlaintextMatrix(original []complex128, r, c int, inRowPacked bool, ouRowPacked bool) ([][]float64, []float64) {
// 	var rid, cid, newi int
// 	mt := make([][]float64, c)
// 	vc := make([]float64, r*c)
// 	for i := 0; i < r*c; i++ {
// 		// 1. compute the normal ind of the element in matrix
// 		if inRowPacked {
// 			rid = i / c
// 			cid = i % c
// 		} else {
// 			rid = i % r
// 			cid = i / r
// 		}

// 		// 2. put this element from (rid, cid) to (cid, rid)
// 		if ouRowPacked {
// 			newi = cid*r + rid
// 		} else {
// 			newi = rid*c + cid
// 		}

// 		vc[newi] = real(original[i])

// 		// mt[newi] = original[i]
// 		if mt[cid] == nil {
// 			mt[cid] = make([]float64, r)
// 		}
// 		mt[cid][rid] = real(original[i])
// 	}
// 	return mt, vc
// }

// func TestMarshallUnmarshall(t *testing.T) {
// 	params := ckks.DefaultParams[ckks.PN14QP438]
// 	fmt.Println()
// 	fmt.Println("=========================================")
// 	fmt.Println("         INSTANTIATING SCHEME            ")
// 	fmt.Println("=========================================")
// 	fmt.Println()

// 	kgen := ckks.NewKeyGenerator(params)
// 	sk := kgen.GenSecretKey()
// 	// rlk := kgen.GenRelinearizationKey(sk)
// 	encryptor := ckks.NewEncryptorFromSk(params, sk)
// 	decryptor := ckks.NewDecryptor(params, sk)
// 	encoder := ckks.NewEncoder(params)

// 	slots := params.Slots()
// 	plainWeights := make([]complex128, slots)
// 	for i, _ := range plainWeights {
// 		if i >= 10 {
// 			break
// 		}
// 		plainWeights[i] = complex(float64(i%5), 0)
// 	}
// 	encodeWeights := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), plainWeights, params.LogSlots())
// 	encryptWeights := encryptor.EncryptNew(encodeWeights)

// 	data, err := encryptWeights.MarshalBinary()
// 	if err != nil {
// 		panic("err in marshall")
// 	}

// 	// var out *ckks.Ciphertext
// 	out := new(ckks.Ciphertext)
// 	err = out.UnmarshalBinary(data)
// 	if err != nil {
// 		panic("err in unmarshall")
// 	}

// 	valuesBefore := encoder.Decode(decryptor.DecryptNew(encryptWeights), params.LogSlots())
// 	valuesAfter := encoder.Decode(decryptor.DecryptNew(out), params.LogSlots())

// 	fmt.Printf("before marshall: %v ...\n", valuesBefore[:10])
// 	fmt.Printf("after marshall: %v ...\n", valuesAfter[:10])

// }

// // TestForward test forward in conv1d and avg pool
// func TestLinearTransform(t *testing.T) {
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
// 	nmakers := 3

// 	// inds := params.RotationsForInnerSumLog(1, 4)
// 	// inds = append(inds, -1)
// 	// fmt.Printf("original inds: %v\n", inds)
// 	// rks := kgen.GenRotationKeysForRotations(inds, false, sk)
// 	// evaluator := ckks.NewEvaluator(params, ckks.EvaluationKey{Rlk: rlk, Rtks: rks})

// 	// create a test input with 2 cells and 4 makers for each cell
// 	values := make([]complex128, slots)
// 	for i := range values {
// 		if i >= ncells*nmakers {
// 			break
// 		}
// 		values[i] = complex(float64(i), 0)
// 	}

// 	plainX := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), values, params.LogSlots())

// 	cipherX := encryptor.EncryptNew(plainX)

// 	fmt.Println("num of slots: ", params.Slots())

// 	r := ncells
// 	c := nmakers
// 	inRowPacked := true
// 	ouRowPacked := true

// 	colsMatrix := utils.GenTransposeMatrix(params.Slots(), r, c, inRowPacked, ouRowPacked)

// 	transposeVec := utils.GenTransposeMap(colsMatrix)

// 	// trsRotInds := GetRotationInds(transposeVec)
// 	// kgen.GenRotationIndexesForDiagMatrix()

// 	diagM := encoder.EncodeDiagMatrixAtLvl(params.MaxLevel(), transposeVec, params.Scale(), 8.0, params.LogSlots())

// 	newInds := kgen.GenRotationIndexesForDiagMatrix(diagM)

// 	fmt.Printf("new rotation keys: %v", newInds)

// 	rks := kgen.GenRotationKeysForRotations(newInds, false, sk)

// 	evaluator := ckks.NewEvaluator(params, ckks.EvaluationKey{Rlk: rlk, Rtks: rks})

// 	transformed := evaluator.LinearTransform(cipherX, diagM)

// 	fmt.Printf("Values if input: %v\n", values[:8])

// 	PrintDebugMatrix(params, transformed[0], values, decryptor, encoder, r, c, inRowPacked, ouRowPacked)
// }

// /*
// I am a little question about the usage of LinearTransform
// For example, given s = params.Slots() = 2^3
// now I want to transpose a matrix:
//  [[0, 1],
//   [2, 3]]

// row packing, the matrix becomes a vector: v1 = [0, 1, 2, 3, 0.....]

// now I generate the transform matrix with size (8*8):
//  [[1, 0, 0, 0...],
//   [0, 0, 1, 0...],
//   [0, 1, 0, 0...],
//   [0, 0, 0, 1...],
//   ...
//   [0, 0, 0, 0...]]

// Now I create the DiagMatrix: len(diag[i]) = 8
// diag[0] = [1,0,0,1,0...]
// diag[1] = [0,1,0,0,0...]
// diag[7] = [0,0,0,1,0...]

// the rotation key i required: [0, -1, -7]

// I test it on this 2*2 matrix, and it works,
// but on larger matrix, it fails because of (switching key not available) or wrong transpose result
// I wonder if I use a wrong key of the diag map (map[int][]complex128)
// Or I miss some rotation keys.

// */
