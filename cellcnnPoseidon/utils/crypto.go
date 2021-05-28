package utils

import (
	"fmt"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/utils"
)

var ThreadsCount = 1

// CipherVector is a slice of Ciphertexts
type CipherVector []*ckks.Ciphertext

// CipherMatrix is a slice of slice of Ciphertexts
type CipherMatrix []CipherVector

// PlainVector is a slice of Plaintexts
type PlainVector []*ckks.Plaintext

// PlainMatrix is a slice of slice of Plaintexts
type PlainMatrix []PlainVector

// CryptoParams aggregates all ckks scheme information
type CryptoParams struct {
	Sk          *ckks.SecretKey
	AggregateSk *ckks.SecretKey
	Pk          *ckks.PublicKey
	Rlk         *ckks.RelinearizationKey
	Kgen        ckks.KeyGenerator
	// RotKs       *ckks.RotationKeys
	Params *ckks.Parameters

	encoders   chan ckks.Encoder
	encryptors chan ckks.Encryptor
	decryptors chan ckks.Decryptor
	evaluators chan ckks.Evaluator
}

// CryptoParamsForNetwork stores all crypto info to save to file
type CryptoParamsForNetwork struct {
	params      *ckks.Parameters
	sk          []*ckks.SecretKey
	aggregateSk *ckks.SecretKey
	pk          *ckks.PublicKey
	// rlk         *ckks.EvaluationKey
	// rotKs       *ckks.RotationKeys
}

// var _ encoding.BinaryMarshaler = new(CryptoParamsForNetwork)
// var _ encoding.BinaryUnmarshaler = new(CryptoParamsForNetwork)

// RotationType defines how much we should rotate and in which direction
// type RotationType struct {
// 	Value int
// 	Side  data.Side
// }

// CKKSParamsForTests are _unsecure_ and fast parameters
// var CKKSParamsForTests = NewCKKSParamsForTests()

// NewCKKSParamsForTests initializes new _unsecure_ fast CryptoParams for testing
// func NewCKKSParamsForTests() *ckks.Parameters {
// 	var CKKSParamsForTests, _ = ckks.NewParametersFromLogModuli(8, &ckks.LogModuli{LogQi: []uint64{36, 30, 30, 30, 30, 30, 30, 30, 30, 30}, LogPi: []uint64{32, 32, 32}})
// 	CKKSParamsForTests.SetScale(1 << 30)
// 	CKKSParamsForTests.SetLogSlots(CKKSParamsForTests.LogN() - 1)
// 	return CKKSParamsForTests
// }

// #------------------------------------#
// #------------ INIT ------------------#
// #------------------------------------#

func NewCryptoPlaceHolder(
	params *ckks.Parameters, kgen ckks.KeyGenerator,
	sk *ckks.SecretKey, rlk *ckks.RelinearizationKey,
	encoder ckks.Encoder, encryptor ckks.Encryptor,
) *CryptoParams {

	encoders := make(chan ckks.Encoder, 1)
	encoders <- encoder

	encryptors := make(chan ckks.Encryptor, 1)
	encryptors <- encryptor

	return &CryptoParams{
		Params:     params,
		Sk:         sk,
		Rlk:        rlk,
		Kgen:       kgen,
		encoders:   encoders,
		encryptors: encryptors,
	}
}

// NewCryptoParams initializes CryptoParams with the given values
func NewCryptoParams(params *ckks.Parameters, kgen ckks.KeyGenerator, sk, aggregateSk *ckks.SecretKey, pk *ckks.PublicKey, rlk *ckks.RelinearizationKey) *CryptoParams {
	evaluators := make(chan ckks.Evaluator, ThreadsCount)
	// for i := 0; i < ThreadsCount; i++ {
	// 	evaluators <- eval.ShallowCopy()
	// }

	encoders := make(chan ckks.Encoder, ThreadsCount)
	for i := 0; i < ThreadsCount; i++ {
		encoders <- ckks.NewEncoder(params)
	}

	encryptors := make(chan ckks.Encryptor, ThreadsCount)
	for i := 0; i < ThreadsCount; i++ {
		encryptors <- ckks.NewEncryptorFromPk(params, pk)
	}

	decryptors := make(chan ckks.Decryptor, ThreadsCount)
	for i := 0; i < ThreadsCount; i++ {
		decryptors <- ckks.NewDecryptor(params, sk)
	}

	return &CryptoParams{
		Params:      params,
		Sk:          sk,
		AggregateSk: aggregateSk,
		Pk:          pk,
		Rlk:         rlk,
		Kgen:        kgen,

		encoders:   encoders,
		encryptors: encryptors,
		decryptors: decryptors,
		evaluators: evaluators,
	}
}

// SetDecryptors sets the decryptors in the CryptoParams object
func (cp *CryptoParams) SetDecryptors(params *ckks.Parameters, sk *ckks.SecretKey) {
	decryptors := make(chan ckks.Decryptor, ThreadsCount)
	for i := 0; i < ThreadsCount; i++ {
		decryptors <- ckks.NewDecryptor(params, sk)
	}
	cp.decryptors = decryptors
}

// SetEncryptors sets the encryptors in the CryptoParams object
func (cp *CryptoParams) SetEncryptors(params *ckks.Parameters, pk *ckks.PublicKey) {
	encryptors := make(chan ckks.Encryptor, ThreadsCount)
	for i := 0; i < ThreadsCount; i++ {
		encryptors <- ckks.NewEncryptorFromPk(params, pk)
	}
	cp.encryptors = encryptors
}

// SetEncryptors sets the encryptors in the CryptoParams object
func (cp *CryptoParams) SetEvaluator(eval ckks.Evaluator) {
	evaluators := make(chan ckks.Evaluator, ThreadsCount)
	for i := 0; i < ThreadsCount; i++ {
		evaluators <- eval.ShallowCopy()
	}
}

// NewCryptoParamsForNetwork initializes a set of nbrNodes CryptoParams each containing: keys, encoder, encryptor, decryptor, etc.
// func NewCryptoParamsForNetwork(params *ckks.Parameters, nbrNodes int) []*CryptoParams {
// 	kgen := ckks.NewKeyGenerator(params)

// 	aggregateSk := ckks.NewSecretKey(params)
// 	skList := make([]*ckks.SecretKey, nbrNodes)
// 	rq, _ := ring.NewRing(params.N(), append(params.Qi(), params.Pi()...))

// 	for i := 0; i < nbrNodes; i++ {
// 		skList[i] = kgen.GenSecretKey()
// 		rq.Add(aggregateSk.Get(), skList[i].Get(), aggregateSk.Get())
// 	}
// 	pk := kgen.GenPublicKey(aggregateSk)

// 	ret := make([]*CryptoParams, nbrNodes)
// 	for i := range ret {
// 		rlk := kgen.GenRelinKey(aggregateSk)
// 		ret[i] = NewCryptoParams(params, skList[i], aggregateSk, pk, rlk)
// 	}
// 	return ret
// }

// NewCryptoParamsFromPath reads given path and return the parsed CryptoParams
// func NewCryptoParamsFromPath(path string) ([]*CryptoParams, error) {
// 	encoded, err := ioutil.ReadFile(path)
// 	if err != nil {
// 		return nil, fmt.Errorf("read file: %w", err)
// 	}

// 	ret := new(CryptoParamsForNetwork)
// 	if err := ret.UnmarshalBinary(encoded); err != nil {
// 		return nil, fmt.Errorf("decode: %v", err)
// 	}

// 	cryptoParamsList := make([]*CryptoParams, len(ret.sk))
// 	for i := range cryptoParamsList {
// 		cryptoParamsList[i] = NewCryptoParams(ret.params, ret.sk[i], ret.aggregateSk, ret.pk, ret.rlk)
// 		cryptoParamsList[i].RotKs = ret.rotKs
// 	}

// 	return cryptoParamsList, nil
// }

// Replicate generates n new cryptoParams to be given to the nodes
// func (cp *CryptoParams) Replicate(nbrNodes int) []*CryptoParams {
// 	ret := make([]*CryptoParams, nbrNodes)
// 	ret[0] = cp
// 	for i := range ret[1:] {
// 		ret[i+1] = NewCryptoParams(cp.Params, cp.Sk, cp.AggregateSk, cp.Pk, cp.Rlk)
// 		ret[i+1].RotKs = cp.RotKs
// 	}
// 	return ret
// }

// // SetRotKeys sets/adds new rotation keys
// func (cp *CryptoParams) SetRotKeys(nbrRot []RotationType) {
// 	rotKeys := ckks.NewRotationKeys()
// 	for _, n := range nbrRot {
// 		GenRot(cp, n.Value, n.Side, rotKeys)
// 	}
// 	cp.RotKs = rotKeys
// }

// GenRot generates a left or right rotation of a ciphertext
// func GenRot(cryptoParams *CryptoParams, rotation int, side data.Side, rotKeys *ckks.RotationKeys) *ckks.RotationKeys {
// 	kgen := ckks.NewKeyGenerator(cryptoParams.Params)
// 	if rotKeys == nil {
// 		rotKeys = ckks.NewRotationKeys()
// 	}

// 	if side == sides.Right {
// 		kgen.GenRotationKey(ckks.RotationLeft, cryptoParams.AggregateSk, uint64(cryptoParams.GetSlots()-rotation), rotKeys)
// 	} else {
// 		kgen.GenRotationKey(ckks.RotationLeft, cryptoParams.AggregateSk, uint64(rotation), rotKeys)
// 	}
// 	return rotKeys
// }

// GenRandSeed generates 64 random bytes
// typically to use as see of the PRNG
// func GenRandSeed() ([]byte, error) {
// 	randomSeed := make([]byte, 64)
// 	_, err := rand.Read(randomSeed)
// 	if err != nil {
// 		return nil, err
// 	}
// 	return randomSeed, nil
// }

// GenCRPWithSeed generates one CRP (common reference polynomial)
// using the given seed to generate the CRP
func GenCRPWithSeed(cryptoParams *CryptoParams, seed []byte) (*ring.Poly, error) {
	crp, err := GenCRPListWithSeed(cryptoParams, seed, 1) // Generate only one crp
	if err != nil || len(crp) != 1 {
		return nil, fmt.Errorf("generating one CRP: %v", err)
	}
	return crp[0], nil
}

// GenCRPListWithSeed generates an array of size n of common reference polynomials
// using the given seed to generate each CRP of the list
func GenCRPListWithSeed(cryptoParams *CryptoParams, randomSeed []byte, n int) ([]*ring.Poly, error) {
	prng, err := utils.NewKeyedPRNG(randomSeed) // Uses a new random seed
	if err != nil {
		return nil, fmt.Errorf("creating PRNG: %v", err)
	}
	sampler, err := NewSampler(cryptoParams, prng)
	if err != nil {
		return nil, fmt.Errorf("creating PRNG uniform sampler: %v", err)
	}

	crpList := make([]*ring.Poly, n)
	for i := range crpList {
		crpList[i] = sampler.ReadNew()
	}
	return crpList, nil
}

// NewSampler creates a uniform sampler from the PRNG and the crypto params
func NewSampler(cryptoParams *CryptoParams, prng utils.PRNG) (*ring.UniformSampler, error) {
	ringQP, err := ring.NewRing(cryptoParams.Params.N(), append(cryptoParams.Params.Qi(), cryptoParams.Params.Pi()...))
	if err != nil {
		return nil, fmt.Errorf("creating new ring: %v", err)
	}

	sampler := ring.NewUniformSampler(prng, ringQP)
	return sampler, nil
}

// GetScaleByLevel returns a scale for a given level
func (cp *CryptoParams) GetScaleByLevel(level uint64) float64 {
	return float64(cp.Params.Qi()[level])
}

// GetSlots gets the number of encodable slots (N/2)
func (cp *CryptoParams) GetSlots() int {
	return 1 << cp.Params.LogSlots()
}

// WithEncoder run the given function with an encoder
func (cp *CryptoParams) WithEncoder(act func(ckks.Encoder) error) error {
	encoder := <-cp.encoders
	err := act(encoder)
	cp.encoders <- encoder
	return err
}

func (cp *CryptoParams) GetEncoder() ckks.Encoder {
	return ckks.NewEncoder(cp.Params)
}

func (cp *CryptoParams) GetEncryptor() ckks.Encryptor {
	// return ckks.NewEncryptorFromPk(cp.Params, cp.Pk)
	// tmp := <-cp.encryptors
	// cp.encryptors <- tmp
	// return tmp
	return ckks.NewEncryptorFromSk(cp.Params, cp.Sk)
}

func (cp *CryptoParams) GetDecryptor() ckks.Decryptor {
	// return ckks.NewEncryptorFromPk(cp.Params, cp.Pk)
	tmp := <-cp.decryptors
	cp.decryptors <- tmp
	return tmp
}

func (cp *CryptoParams) GetEvaluator() ckks.Evaluator {
	tmp := <-cp.evaluators
	cp.evaluators <- tmp
	return tmp
}

// WithEncryptor run the given function with an encryptor
func (cp *CryptoParams) WithEncryptor(act func(ckks.Encryptor) error) error {
	encryptor := <-cp.encryptors
	err := act(encryptor)
	cp.encryptors <- encryptor
	return err
}

// WithDecryptor run the given function with a decryptor
func (cp *CryptoParams) WithDecryptor(act func(act ckks.Decryptor) error) error {
	decryptor := <-cp.decryptors
	err := act(decryptor)
	cp.decryptors <- decryptor
	return err
}

// WithEvaluator run the given function with an evaluator
func (cp *CryptoParams) WithEvaluator(act func(ckks.Evaluator) error) error {
	eval := <-cp.evaluators
	err := act(eval)
	cp.evaluators <- eval
	return err
}

// LevelTest tests if a ciphertext level is higher or equal to a target level. If not we need to bootstrap.
// func (cp *CryptoParams) LevelTest(cv *ckks.Ciphertext, targetLevel uint64) bool {
// 	// log2(ct.Scale() * 2^128) gives the minimum modulus we need for a safe bootstrap
// 	minMod := math.Log2(cv.Scale() * math.Pow(2, 128))
// 	currMod := 0.0
// 	bootstrapThreshold := uint64(0)
// 	for i := range cp.Params.Qi() {
// 		currMod += math.Log2(float64(cp.Params.Qi()[i]))
// 		if currMod > minMod {
// 			break
// 		}
// 		bootstrapThreshold++
// 	}

// 	if cp.Params.MaxLevel() < targetLevel+bootstrapThreshold {
// 		panic("There are not enough levels to perform the next operation. Insecure bootstrap!")
// 	} else if cv.Level() < targetLevel+bootstrapThreshold {
// 		return true
// 	} else {
// 		return false
// 	}
// }

// #------------------------------------#
// #------------ ENCRYPTION ------------#
// #------------------------------------#

// NewEncryptedMatrix initializes a matrix with dx and dy dimensions with 0's
// func NewEncryptedMatrix(cryptoParams *CryptoParams, dy int, dx int) ([]CipherVector, int, int, error) {
// 	matrix := make([][]float64, dy)
// 	for i := range matrix {
// 		matrix[i] = make([]float64, dx)
// 	}
// 	return EncryptFloatMatrixRow(cryptoParams, matrix)
// }

// // NewEncryptedMatrixWith initializes a matrix with dx and dy dimensions with x
// func NewEncryptedMatrixWith(cryptoParams *CryptoParams, dy int, dx int, x float64) ([]CipherVector, int, int, error) {
// 	matrix := make([][]float64, dy)
// 	for i := range matrix {
// 		matrix[i] = make([]float64, dx)
// 		for j := range matrix[0] {
// 			matrix[i][j] = x
// 		}
// 	}
// 	return EncryptFloatMatrixRow(cryptoParams, matrix)
// }

// // EncryptFloat encrypts one float64 value.
// func EncryptFloat(cryptoParams *CryptoParams, num float64) *ckks.Ciphertext {
// 	N := cryptoParams.GetSlots()
// 	plaintext := ckks.NewPlaintext(cryptoParams.Params, cryptoParams.Params.MaxLevel(), cryptoParams.Params.Scale())

// 	cryptoParams.WithEncoder(func(encoder ckks.Encoder) error {
// 		encoder.Encode(plaintext, ConvertVectorFloat64ToComplex(PadVector([]float64{num}, N)), uint64(N))
// 		return nil
// 	})

// 	var ciphertext *ckks.Ciphertext
// 	cryptoParams.WithEncryptor(func(encryptor ckks.Encryptor) error {
// 		ciphertext = encryptor.EncryptNew(plaintext)
// 		return nil
// 	})
// 	return ciphertext
// }

// EncryptFloatVector encrypts a slice of float64 values in multiple batched ciphertexts.
// and return the number of encrypted elements.
// func EncryptFloatVector(cryptoParams *CryptoParams, f []float64) (CipherVector, int) {
// 	nbrMaxCoef := cryptoParams.GetSlots()
// 	length := len(f)

// 	cipherArr := make(CipherVector, 0)
// 	elementsEncrypted := 0
// 	for elementsEncrypted < length {
// 		start := elementsEncrypted
// 		end := elementsEncrypted + nbrMaxCoef

// 		if end > length {
// 			end = length
// 		}
// 		plaintext := ckks.NewPlaintext(cryptoParams.Params, cryptoParams.Params.MaxLevel(), cryptoParams.Params.Scale())
// 		// pad to 0s
// 		cryptoParams.WithEncoder(func(encoder ckks.Encoder) error {
// 			encoder.Encode(plaintext, ConvertVectorFloat64ToComplex(PadVector(f[start:end], nbrMaxCoef)), uint64(nbrMaxCoef))
// 			return nil
// 		})
// 		var cipher *ckks.Ciphertext
// 		cryptoParams.WithEncryptor(func(encryptor ckks.Encryptor) error {
// 			cipher = encryptor.EncryptNew(plaintext)
// 			return nil
// 		})
// 		cipherArr = append(cipherArr, cipher)
// 		elementsEncrypted = elementsEncrypted + (end - start)
// 	}
// 	return cipherArr, elementsEncrypted
// }

// EncryptFloatMatrixRow encrypts a matrix of float64 to multiple packed ciphertexts.
// For this specific matrix encryption each row is encrypted in a set of ciphertexts.
// func EncryptFloatMatrixRow(cryptoParams *CryptoParams, matrix [][]float64) (CipherMatrix, int, int, error) {
// 	nbrRows := len(matrix)
// 	d := len(matrix[0])

// 	matrixEnc := make([]CipherVector, 0)
// 	for _, row := range matrix {
// 		if d != len(row) {
// 			return nil, 0, 0, errors.New("this is not a matrix (expected " + strconv.FormatInt(int64(d), 10) +
// 				" dimensions but got " + strconv.FormatInt(int64(len(row)), 10))
// 		}
// 		rowEnc, _ := EncryptFloatVector(cryptoParams, row)
// 		matrixEnc = append(matrixEnc, rowEnc)
// 	}
// 	return matrixEnc, nbrRows, d, nil
// }

// // EncodeFloatVector encodes a slice of float64 values in multiple batched plaintext (ready to be encrypted).
// // It also returns the number of encoded elements.
// func EncodeFloatVector(cryptoParams *CryptoParams, f []float64, level uint64, scale float64) (PlainVector, int) {
// 	nbrMaxCoef := cryptoParams.GetSlots()
// 	length := len(f)

// 	plainArr := make(PlainVector, 0)
// 	elementsEncoded := 0
// 	for elementsEncoded < length {
// 		start := elementsEncoded
// 		end := elementsEncoded + nbrMaxCoef

// 		if end > length {
// 			end = length
// 		}
// 		plaintext := ckks.NewPlaintext(cryptoParams.Params, level, scale)
// 		cryptoParams.WithEncoder(func(encoder ckks.Encoder) error {
// 			encoder.EncodeNTT(plaintext, ConvertVectorFloat64ToComplex(PadVector(f[start:end], nbrMaxCoef)), uint64(nbrMaxCoef))
// 			return nil
// 		})
// 		plainArr = append(plainArr, plaintext)
// 		elementsEncoded = elementsEncoded + (end - start)
// 	}
// 	return plainArr, elementsEncoded
// }

// // EncodeFloatMatrixRow encodes a matrix of float64 to multiple packed plaintexts.
// // For this specific matrix encoding each row is encoded in a set of plaintexts.
// func EncodeFloatMatrixRow(cryptoParams *CryptoParams, matrix [][]float64, level uint64, scale float64) (PlainMatrix, int, int, error) {
// 	nbrRows := len(matrix)
// 	d := len(matrix[0])

// 	matrixEnc := make(PlainMatrix, 0)
// 	for _, row := range matrix {
// 		if d != len(row) {
// 			return nil, 0, 0, errors.New("this is not a matrix (expected " + strconv.FormatInt(int64(d), 10) +
// 				" dimensions but got " + strconv.FormatInt(int64(len(row)), 10))
// 		}

// 		rowEnc, _ := EncodeFloatVector(cryptoParams, row, level, scale)
// 		matrixEnc = append(matrixEnc, rowEnc)
// 	}
// 	return matrixEnc, nbrRows, d, nil
// }

// // NewMask creates a batched plaintext that can be used to mask an existing batched ciphertext.
// // In other words is allows us to keep only certain coefficients from the ciphertext.
// func NewMask(cryptoParams *CryptoParams, maskClear []float64, level uint64, scale float64) *ckks.Plaintext {
// 	mask := ckks.NewPlaintext(cryptoParams.Params, level, scale)
// 	cryptoParams.WithEncoder(func(encoder ckks.Encoder) error {
// 		encoder.EncodeNTT(mask, ConvertVectorFloat64ToComplex(maskClear), uint64(cryptoParams.GetSlots()))
// 		return nil
// 	})
// 	return mask
// }

// // NewOneValueMask creates a batched plaintext with 1 value in a single position that can be used to mask an existing batched ciphertext.
// func NewOneValueMask(cryptoParams *CryptoParams, index int, value float64, level uint64, scale float64) *ckks.Plaintext {
// 	maskClear := make([]float64, cryptoParams.GetSlots())
// 	maskClear[index] = value
// 	mask := ckks.NewPlaintext(cryptoParams.Params, level, scale)
// 	cryptoParams.WithEncoder(func(encoder ckks.Encoder) error {
// 		encoder.EncodeNTT(mask, ConvertVectorFloat64ToComplex(PadVector(maskClear, cryptoParams.GetSlots())), uint64(cryptoParams.GetSlots()))
// 		return nil
// 	})
// 	return mask
// }

// // #------------------------------------#
// // #------------ DECRYPTION ------------#
// // #------------------------------------#

// // DecryptFloat decrypts a ciphertext with one float64 value.
// func DecryptFloat(cryptoParams *CryptoParams, cipher *ckks.Ciphertext) float64 {
// 	var ret float64
// 	var plaintext *ckks.Plaintext

// 	cryptoParams.WithDecryptor(func(decryptor ckks.Decryptor) error {
// 		plaintext = decryptor.DecryptNew(cipher)
// 		return nil
// 	})
// 	cryptoParams.WithEncoder(func(encoder ckks.Encoder) error {
// 		ret = real(encoder.Decode(plaintext, uint64(cryptoParams.GetSlots()))[0])
// 		return nil
// 	})

// 	return ret
// }

// // DecryptMultipleFloat decrypts a ciphertext with multiple float64 values.
// // If nbrEl<=0 it decrypts everything without caring about the number of encrypted values.
// // If nbrEl>0 the function returns N elements from the decryption.
// func DecryptMultipleFloat(cryptoParams *CryptoParams, cipher *ckks.Ciphertext, nbrEl int) []float64 {
// 	var plaintext *ckks.Plaintext

// 	cryptoParams.WithDecryptor(func(decryptor ckks.Decryptor) error {
// 		plaintext = decryptor.DecryptNew(cipher)
// 		return nil
// 	})

// 	var val []complex128
// 	cryptoParams.WithEncoder(func(encoder ckks.Encoder) error {
// 		val = encoder.Decode(plaintext, uint64(cryptoParams.GetSlots()))
// 		return nil
// 	})
// 	dataDecrypted := ConvertVectorComplexToFloat64(val)
// 	if nbrEl <= 0 {
// 		return dataDecrypted
// 	}
// 	return dataDecrypted[:nbrEl]
// }

// // DecryptFloatVector decrypts multiple batched ciphertexts with N float64 values and appends
// // all data into one single float vector.
// // If nbrEl<=0 it decrypts everything without caring about the number of encrypted values.
// // If nbrEl>0 the function returns N elements from the decryption.
// func DecryptFloatVector(cryptoParams *CryptoParams, fEnc CipherVector, N int) []float64 {
// 	var plaintext *ckks.Plaintext

// 	dataDecrypted := make([]float64, 0)
// 	for _, cipher := range fEnc {
// 		cryptoParams.WithDecryptor(func(decryptor ckks.Decryptor) error {
// 			plaintext = decryptor.DecryptNew(cipher)
// 			return nil
// 		})
// 		var val []complex128
// 		cryptoParams.WithEncoder(func(encoder ckks.Encoder) error {
// 			val = encoder.Decode(plaintext, uint64(cryptoParams.GetSlots()))
// 			return nil
// 		})
// 		dataDecrypted = append(dataDecrypted, ConvertVectorComplexToFloat64(val)...)
// 	}
// 	if N <= 0 {
// 		return dataDecrypted
// 	}
// 	return dataDecrypted[:N]
// }

// // DecryptFloatVectorIndep decrypts multiple batched ciphertexts with N float64 values,
// // placing each chunk of data in a different/independent float vector.
// // If nbrEl<=0 it decrypts everything without caring about the number of encrypted values.
// // If nbrEl>0 the function returns N elements from the decryption.
// func DecryptFloatVectorIndep(cryptoParams *CryptoParams, fEnc CipherVector, N int) [][]float64 {
// 	var plaintext *ckks.Plaintext

// 	dataDecrypted := make([][]float64, len(fEnc))
// 	for i, cipher := range fEnc {
// 		cryptoParams.WithDecryptor(func(decryptor ckks.Decryptor) error {
// 			plaintext = decryptor.DecryptNew(cipher)
// 			return nil
// 		})
// 		var val []complex128
// 		cryptoParams.WithEncoder(func(encoder ckks.Encoder) error {
// 			val = encoder.Decode(plaintext, uint64(cryptoParams.GetSlots()))
// 			return nil
// 		})
// 		dataDecrypted[i] = ConvertVectorComplexToFloat64(val)

// 		if N <= 0 {
// 			dataDecrypted[i] = ConvertVectorComplexToFloat64(val)
// 		} else {
// 			dataDecrypted[i] = ConvertVectorComplexToFloat64(val)[:N]
// 		}
// 	}
// 	return dataDecrypted
// }

// // DecryptFloatMatrix decrypts a matrix (kind of) of multiple packed ciphertexts.
// // For this specific matrix decryption each row is encrypted in a set of ciphertexts.
// // d is the number of column values
// func DecryptFloatMatrix(cryptoParams *CryptoParams, matrixEnc []CipherVector, d int) [][]float64 {
// 	matrix := make([][]float64, 0)
// 	for _, rowEnc := range matrixEnc {
// 		row := DecryptFloatVector(cryptoParams, rowEnc, d)
// 		matrix = append(matrix, row)
// 	}
// 	return matrix
// }

// // DecodeFloatVector decodes a slice of plaintext values in multiple float64 values.
// func DecodeFloatVector(cryptoParams *CryptoParams, fEncoded PlainVector, elementsEncoded int) []float64 {
// 	dataDecoded := make([]float64, 0)
// 	for _, plaintext := range fEncoded {
// 		var val []complex128
// 		cryptoParams.WithEncoder(func(encoder ckks.Encoder) error {
// 			val = encoder.Decode(plaintext, uint64(cryptoParams.GetSlots()))
// 			return nil
// 		})
// 		dataDecoded = append(dataDecoded, ConvertVectorComplexToFloat64(val)...)
// 	}
// 	return dataDecoded
// }

// // #------------------------------------#
// // #------------ MARSHALL --------------#
// // #------------------------------------#

// // MarshalBinary -> CipherMatrix: converts a matrix of ciphertexts to an array of bytes.
// // Returns the number of rows and the number of ciphertexts per row
// func (cm *CipherMatrix) MarshalBinary() ([]byte, [][]int, error) {
// 	b := make([]byte, 0)
// 	ctSizes := make([][]int, len(*cm))
// 	for i, v := range *cm {
// 		tmp, n, err := v.MarshalBinary()
// 		ctSizes[i] = n
// 		if err != nil {
// 			return nil, nil, err
// 		}
// 		b = append(b, tmp...)
// 	}

// 	return b, ctSizes, nil

// }

// // UnmarshalBinary -> CipherMatrix: converts an array of bytes to an matrix of ciphertexts.
// func (cm *CipherMatrix) UnmarshalBinary(cryptoParams *CryptoParams, f []byte, ctSizes [][]int) error {
// 	*cm = make([]CipherVector, len(ctSizes))

// 	start := 0
// 	for i := range ctSizes {
// 		rowSize := 0
// 		for j := range ctSizes[i] {
// 			rowSize += ctSizes[i][j]
// 		}
// 		end := start + rowSize
// 		cv := make(CipherVector, 0)
// 		err := cv.UnmarshalBinary(cryptoParams, f[start:end], ctSizes[i])
// 		if err != nil {
// 			return err
// 		}
// 		start = end
// 		(*cm)[i] = cv
// 	}
// 	return nil
// }

// // MarshalBinary -> CipherVector: converts an array of ciphertexts to an array of bytes.
// // Return the original number of ciphertexts
// func (cv *CipherVector) MarshalBinary() ([]byte, []int, error) {
// 	data := make([]byte, 0)
// 	ctSizes := make([]int, 0)
// 	for _, ct := range *cv {
// 		b, err := ct.MarshalBinary()
// 		if err != nil {
// 			return nil, nil, err
// 		}
// 		data = append(data, b...)
// 		ctSizes = append(ctSizes, len(b))
// 	}
// 	return data, ctSizes, nil
// }

// // UnmarshalBinary -> CipherVector: converts an array of bytes to an array of ciphertexts.
// func (cv *CipherVector) UnmarshalBinary(cryptoParams *CryptoParams, f []byte, fSizes []int) error {
// 	*cv = make(CipherVector, len(fSizes))

// 	start := 0
// 	for i := 0; i < len(fSizes); i++ {
// 		ct := ckks.NewCiphertext(cryptoParams.Params, 1, cryptoParams.Params.MaxLevel(), cryptoParams.Params.Scale())
// 		if err := ct.UnmarshalBinary(f[start : start+fSizes[i]]); err != nil {
// 			return err
// 		}
// 		(*cv)[i] = ct
// 		start += fSizes[i]
// 	}
// 	return nil
// }

// type cryptoParamsMarshalable struct {
// 	Params      *ckks.Parameters
// 	Sk          []*ckks.SecretKey
// 	AggregateSk *ckks.SecretKey
// 	Pk          *ckks.PublicKey
// 	Rlk         *ckks.EvaluationKey
// 	RotKs       *ckks.RotationKeys
// }

// // MarshalBinary encode into a []byte accepted by UnmarshalBinary
// func (cp *CryptoParamsForNetwork) MarshalBinary() ([]byte, error) {
// 	var ret bytes.Buffer
// 	encoder := gob.NewEncoder(&ret)

// 	err := encoder.Encode(cryptoParamsMarshalable{
// 		Params:      cp.params,
// 		Sk:          cp.sk,
// 		AggregateSk: cp.aggregateSk,
// 		Pk:          cp.pk,
// 		Rlk:         cp.rlk,
// 		RotKs:       cp.rotKs,
// 	})
// 	if err != nil {
// 		return nil, fmt.Errorf("encode minimal crypto params: %v", err)
// 	}

// 	return ret.Bytes(), nil
// }

// // UnmarshalBinary decode a []byte created by MarshalBinary
// func (cp *CryptoParamsForNetwork) UnmarshalBinary(data []byte) error {
// 	decoder := gob.NewDecoder(bytes.NewBuffer(data))

// 	decodeParams := new(cryptoParamsMarshalable)
// 	if err := decoder.Decode(decodeParams); err != nil {
// 		return fmt.Errorf("decode minimal crypto params: %v", err)
// 	}

// 	cp.params = decodeParams.Params
// 	cp.sk = decodeParams.Sk
// 	cp.aggregateSk = decodeParams.AggregateSk
// 	cp.pk = decodeParams.Pk
// 	cp.rlk = decodeParams.Rlk
// 	cp.rotKs = decodeParams.RotKs

// 	return nil
// }

// // WriteParamsToFile writes the crypto params to a toml file including the keys' filenames
// func (cp *CryptoParams) WriteParamsToFile(path string, secretKeysList []*ckks.SecretKey) error {
// 	if err := os.MkdirAll(filepath.Dir(path), os.ModePerm); err != nil {
// 		return fmt.Errorf("creating parents dirs of %v: %w", path, err)
// 	}

// 	cpSave := &CryptoParamsForNetwork{
// 		sk:          secretKeysList,
// 		aggregateSk: cp.AggregateSk,
// 		pk:          cp.Pk,
// 		rlk:         cp.Rlk,
// 		rotKs:       cp.RotKs,
// 		params:      cp.Params,
// 	}

// 	encoded, err := cpSave.MarshalBinary()
// 	if err != nil {
// 		return fmt.Errorf("encode: %v", err)
// 	}

// 	if err := ioutil.WriteFile(path, encoded, 0666); err != nil {
// 		return fmt.Errorf("write file: %v", err)
// 	}

// 	return nil
// }

// // #------------------------------------#
// // #-------------- COPY ----------------#
// // #------------------------------------#

// // CopyEncryptedVector does a copy of an array of ciphertexts to a newly created array
// func CopyEncryptedVector(src CipherVector) CipherVector {
// 	dest := make(CipherVector, len(src))
// 	for i := 0; i < len(src); i++ {
// 		dest[i] = (*src[i]).CopyNew().Ciphertext()
// 	}
// 	return dest
// }

// // CopyEncryptedMatrix does a copy of a matrix of ciphertexts to a newly created array
// func CopyEncryptedMatrix(src []CipherVector) []CipherVector {
// 	dest := make([]CipherVector, len(src))
// 	for i := 0; i < len(src); i++ {
// 		dest[i] = CopyEncryptedVector(src[i])
// 	}
// 	return dest
// }

// // ReplicateCiphertext replicates a cipher xn times (each time doubling the nbrElements contained inside the cipher).
// // eg. n=2: [1, 2, 3] -> [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
// func ReplicateCiphertext(cryptoParams *CryptoParams, src *ckks.Ciphertext, nbrElements, n int) (*ckks.Ciphertext, error) {
// 	srcCpy := src.CopyNew().Ciphertext()

// 	for i := 0; i < n; i++ {
// 		cryptoParams.WithEvaluator(func(eval ckks.Evaluator) error {
// 			rt := eval.RotateColumnsNew(srcCpy, uint64(cryptoParams.GetSlots()-nbrElements), cryptoParams.RotKs)
// 			srcCpy = eval.AddNew(srcCpy, rt)
// 			return nil
// 		})

// 		//double the number of elements to replicate
// 		nbrElements *= 2
// 	}
// 	return srcCpy, nil
// }

// // #------------------------------------------#
// // #-------------- OPERATIONS ----------------#
// // #------------------------------------------#

// // RotateForInnerSum performs an inner sum of a ciphertext using rotations and additions
// func RotateForInnerSum(cryptoParams *CryptoParams, ct *ckks.Ciphertext, rotateStart int, size int) *ckks.Ciphertext {
// 	// rotate each one to left size times and add, INNERSUM (rotate i,2i,4i,... times)

// 	//to prevent unnecessary allocation
// 	var temp *ckks.Ciphertext
// 	if int(math.Ceil(math.Log2(float64(size)))) != int(math.Floor(math.Log2(float64(size)))) {
// 		temp = ct.CopyNew().Ciphertext()
// 	}
// 	rotate := rotateStart
// 	for j := 0; j < int(math.Floor(math.Log2(float64(size)))); j++ {
// 		cryptoParams.WithEvaluator(func(eval ckks.Evaluator) error {
// 			rt := eval.RotateColumnsNew(ct, uint64(rotate), cryptoParams.RotKs)
// 			ct = eval.AddNew(ct, rt)
// 			return nil
// 		})
// 		rotate = rotate * 2
// 	}
// 	//if NInputs are not a power of two rotate remaining iteratively until it reaches pow of 2
// 	if int(math.Ceil(math.Log2(float64(size)))) != int(math.Floor(math.Log2(float64(size)))) {
// 		pow := int(math.Floor(math.Log2(float64(size))))
// 		remaining := float64(size) - math.Pow(2, float64(pow))
// 		totalRotated := int(math.Pow(2, float64(pow))) * rotateStart
// 		for j := 0; j < int(remaining); j++ {
// 			cryptoParams.WithEvaluator(func(eval ckks.Evaluator) error {
// 				rotTemp := eval.RotateColumnsNew(temp, uint64(totalRotated), cryptoParams.RotKs).Ciphertext()
// 				ct = eval.AddNew(ct, rotTemp)
// 				return nil
// 			})
// 			totalRotated += rotateStart
// 		}
// 	}
// 	return ct
// }

// // RotateForReplication replicates the contents of a ciphertext numRep times (using rotations)
// func RotateForReplication(cryptoParams *CryptoParams, ct *ckks.Ciphertext, numSlots int, numRep int) *ckks.Ciphertext {
// 	rotate := cryptoParams.GetSlots() - numSlots
// 	//to prevent unnecessary allocation
// 	var temp *ckks.Ciphertext
// 	if int(math.Ceil(math.Log2(float64(numRep)))) != int(math.Floor(math.Log2(float64(numRep)))) {
// 		temp = ct.CopyNew().Ciphertext()
// 	}
// 	for j := 0; j < int(math.Floor(math.Log2(float64(numRep)))); j++ {
// 		cryptoParams.WithEvaluator(func(eval ckks.Evaluator) error {
// 			rt := eval.RotateColumnsNew(ct, uint64(rotate), cryptoParams.RotKs)
// 			ct = eval.AddNew(ct, rt)
// 			return nil
// 		})
// 		rotate = rotate * 2
// 	}

// 	//if NInputs are not a power of two rotate remaining iteratively until it reaches pow of 2
// 	if int(math.Ceil(math.Log2(float64(numRep)))) != int(math.Floor(math.Log2(float64(numRep)))) {
// 		pow := int(math.Floor(math.Log2(float64(numRep))))
// 		remaining := float64(numRep) - math.Pow(2, float64(pow))
// 		totalRotated := int(int(math.Pow(2, float64(pow))) * numSlots)
// 		for j := 0; j < int(remaining); j++ {
// 			cryptoParams.WithEvaluator(func(eval ckks.Evaluator) error {
// 				rotTemp := eval.RotateColumnsNew(temp, uint64(cryptoParams.GetSlots()-totalRotated), cryptoParams.RotKs).Ciphertext()
// 				ct = eval.AddNew(ct, rotTemp)
// 				return nil
// 			})
// 			totalRotated += numSlots
// 		}
// 	}
// 	return ct
// }

// // InnerSumSimpleEnc performs a since inner sum with log(n) rotations and returns the results (position 0 of output ciphertext)
// func InnerSumSimpleEnc(cryptoParams *CryptoParams, ct *ckks.Ciphertext, nbrRotations int, precomputedRotks bool) *ckks.Ciphertext {
// 	var rotKeys *ckks.RotationKeys

// 	rotate := 1
// 	for i := 0; i < nbrRotations; i++ {
// 		if !precomputedRotks {
// 			rotKeys = GenRot(cryptoParams, rotate, false, nil)
// 		} else {
// 			rotKeys = cryptoParams.RotKs
// 		}
// 		cryptoParams.WithEvaluator(func(eval ckks.Evaluator) error {
// 			rt := eval.RotateColumnsNew(ct, uint64(rotate), rotKeys)
// 			ct = eval.AddNew(ct, rt).Ciphertext()
// 			return nil
// 		})
// 		rotate = rotate * 2
// 	}
// 	return ct
// }

// // ComputeN1N2 computes N1 and N2 for the diagonal matrix multiplication (see paper for reference)
// func ComputeN1N2(N int) (int, int) {
// 	N2 := int(math.Floor(math.Sqrt(float64(N))))
// 	N1 := N / N2
// 	return N1, N2
// }

// // MulByDiagMatrix expects the weights encrypted and previously 'replicated' and the encoded diagonal matrix (without 0s columns).
// // We assume the number of features to be able to fit in one single ciphertext. If precompute is true rotations are precomputed.
// func MulByDiagMatrix(nodeName string, cryptoParams *CryptoParams, v *ckks.Ciphertext, mDiag [][]float64, mDiagEncoded PlainMatrix, N1, N2 int, precomputeKeys, precompute bool) (*ckks.Ciphertext, error) {
// 	wRotated := make(map[RotationType]*ckks.Ciphertext)

// 	// Pre-computation of rotated ciphertext
// 	mutex := sync.Mutex{}
// 	wg := sync.WaitGroup{}
// 	wg.Add(N1)
// 	rph := NewRoutinePanicHandler(N1)

// 	for i := 0; i < N1; i++ {
// 		//iT := i
// 		go func(iT int) {
// 			defer wg.Done()
// 			defer rph.Recover(iT)

// 			var rt *ckks.Ciphertext
// 			if iT != 0 {
// 				rotKey := ckks.NewRotationKeys()
// 				if precomputeKeys {
// 					rotKey = cryptoParams.RotKs
// 				} else {
// 					kgen := ckks.NewKeyGenerator(cryptoParams.Params)
// 					kgen.GenRotationKey(ckks.RotationLeft, cryptoParams.Sk, uint64(iT), rotKey)
// 				}
// 				if rotKey == nil {
// 					log.Panic("No rotation key:", iT)
// 				}

// 				// rotate ciphertext left by rot
// 				cryptoParams.WithEvaluator(func(eval ckks.Evaluator) error {
// 					rt = eval.RotateColumnsNew(v, uint64(iT), rotKey)
// 					return nil
// 				})
// 			} else {
// 				rt = v
// 			}
// 			mutex.Lock()
// 			wRotated[RotationType{Value: iT, Side: sides.Left}] = rt.Ciphertext()
// 			mutex.Unlock()
// 		}(i)
// 	}
// 	wg.Wait()
// 	rph.Propagate()

// 	res := ckks.NewCiphertext(cryptoParams.Params, 1, cryptoParams.Params.MaxLevel(), cryptoParams.Params.Scale())
// 	emptyCT := ckks.NewCiphertext(cryptoParams.Params, 1, cryptoParams.Params.MaxLevel(), cryptoParams.Params.Scale())

// 	if !precompute {
// 		// get a weight's ciphertext level and scale
// 		var targetLevel uint64
// 		var targetScale float64
// 		for i := range wRotated {
// 			targetLevel = wRotated[i].Level()
// 			targetScale = wRotated[i].Scale()
// 			break
// 		}

// 		// pre-generate cleartext rotations
// 		mDiagEncoded = make(PlainMatrix, N2)
// 		for j := 0; j < N2; j++ {
// 			mDiagEncoded[j] = make(PlainVector, N1)
// 			for i := 0; i < N1; i++ {
// 				aux := RotateVector(mDiag[N1*j+i], N1*j, true)
// 				tmp, _ := EncodeFloatVector(cryptoParams, aux, targetLevel, targetScale)
// 				mDiagEncoded[j][i] = tmp[0]
// 			}
// 		}
// 	} else {
// 		if mDiagEncoded == nil {
// 			err := errors.New("no precomputed matrix")
// 			log.Error(err)
// 			return nil, err
// 		}
// 	}

// 	mutex2 := sync.Mutex{}
// 	wg2 := sync.WaitGroup{}
// 	wg2.Add(N2)
// 	rph = NewRoutinePanicHandler(N2)
// 	for j := 0; j < N2; j++ {
// 		//jT := j
// 		go func(jT int) {
// 			defer wg2.Done()
// 			defer rph.Recover(jT)

// 			//Temporary ciphertext
// 			tmpVec := emptyCT.CopyNew().Ciphertext()

// 			mutex1 := sync.Mutex{}
// 			wg1 := sync.WaitGroup{}
// 			wg1.Add(N1)
// 			innerRph := NewRoutinePanicHandler(N1)
// 			for i := 0; i < N1; i++ {
// 				go func(iT int, jTT int) {
// 					defer wg1.Done()
// 					defer innerRph.Recover(iT)

// 					w := wRotated[RotationType{
// 						Value: iT,
// 						Side:  sides.Left,
// 					}]

// 					cryptoParams.WithEvaluator(func(eval ckks.Evaluator) error {
// 						tmp := eval.MulRelinNew(w, mDiagEncoded[jTT][iT], cryptoParams.Rlk)

// 						mutex1.Lock()
// 						eval.Add(tmpVec, tmp, tmpVec)
// 						mutex1.Unlock()

// 						return nil
// 					})
// 				}(i, jT)
// 			}
// 			wg1.Wait()
// 			innerRph.Propagate()

// 			cryptoParams.WithEvaluator(func(eval ckks.Evaluator) error {
// 				var rt *ckks.Ciphertext
// 				if jT != 0 {
// 					//Rotation of temporary ciphertext, and addition to result
// 					rotKey := ckks.NewRotationKeys()
// 					if precomputeKeys {
// 						rotKey = cryptoParams.RotKs
// 					} else {
// 						kgen := ckks.NewKeyGenerator(cryptoParams.Params)
// 						kgen.GenRotationKey(ckks.RotationLeft, cryptoParams.Sk, uint64(N1*jT), rotKey)
// 					}
// 					if rotKey == nil {
// 						log.Panic("No rotation key:", N1*jT)
// 					}
// 					rt = eval.RotateColumnsNew(tmpVec, uint64(N1*jT), rotKey)
// 				} else {
// 					rt = tmpVec
// 				}
// 				mutex2.Lock()
// 				eval.Add(res, rt, res)
// 				mutex2.Unlock()

// 				return nil
// 			})
// 		}(j)
// 	}
// 	wg2.Wait()
// 	rph.Propagate()

// 	return res, nil
// }

// // DummyBootstrapping mimics the bootstrapping
// func (cv *CipherVector) DummyBootstrapping(cryptoParams *CryptoParams) CipherVector {
// 	for i, ct := range *cv {
// 		decryptedValues := DecryptMultipleFloat(cryptoParams, ct, 0)
// 		encryptedValues, _ := EncryptFloatVector(cryptoParams, decryptedValues)
// 		(*cv)[i] = encryptedValues[0]
// 	}
// 	return *cv
// }

// // RescaleMatrix rescales an entire matrix of ciphertexts
// func RescaleMatrix(cryptoParams *CryptoParams, src []CipherVector) error {
// 	return cryptoParams.WithEvaluator(func(eval ckks.Evaluator) error {
// 		for _, ctList := range src {
// 			for _, ct := range ctList {
// 				err := eval.Rescale(ct, cryptoParams.Params.Scale(), ct)
// 				if err != nil {
// 					return err
// 				}
// 			}
// 		}

// 		return nil
// 	})
// }

// #------------------------------------#
// #------------- MARSHAL --------------#
// #------------------------------------#

// ArrayPolyToBytes marshals a polynomial into an array of bytes
// func ArrayPolyToBytes(data []*ring.Poly) ([][]byte, error) {
// 	bpolyArr := make([][]byte, len(data))
// 	for i, poly := range data {
// 		bpoly, err := poly.MarshalBinary()
// 		if err != nil {
// 			return nil, fmt.Errorf("marshal poly: %v", err)
// 		}
// 		bpolyArr[i] = bpoly
// 	}

// 	return bpolyArr, nil
// }

// // ArrayPolyFromBytes unmarshals an array of bytes to a polynomial
// func ArrayPolyFromBytes(dataB [][]byte) ([]*ring.Poly, error) {
// 	polyArr := make([]*ring.Poly, len(dataB))
// 	for i, data := range dataB {
// 		poly := new(ring.Poly)
// 		if err := poly.UnmarshalBinary(data); err != nil {
// 			return nil, fmt.Errorf("unmarshal poly: %v", err)
// 		}
// 		polyArr[i] = poly
// 	}

// 	return polyArr, nil
// }
