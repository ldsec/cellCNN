package utils

import (
	"fmt"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

// ThreadsCount num of thread used
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
	Sk          *rlwe.SecretKey
	AggregateSk *rlwe.SecretKey
	Pk          *rlwe.PublicKey
	Rlk         *rlwe.RelinearizationKey
	// Kgen        ckks.KeyGenerator
	// RotKs       *ckks.RotationKeys
	Params ckks.Parameters

	encoders   chan ckks.Encoder
	encryptors chan ckks.Encryptor
	decryptors chan ckks.Decryptor
	evaluators chan ckks.Evaluator
}

// CryptoParamsForNetwork stores all crypto info to save to file
type CryptoParamsForNetwork struct {
	params      ckks.Parameters
	sk          []*rlwe.SecretKey
	aggregateSk *rlwe.SecretKey
	pk          *rlwe.PublicKey
	// rlk         *ckks.EvaluationKey
	// rotKs       *ckks.RotationKeys
}

// #------------------------------------#
// #------------ INIT ------------------#
// #------------------------------------#

// NewCryptoPlaceHolder for debug use, just help pass some parameters
func NewCryptoPlaceHolder(
	params ckks.Parameters,
	sk *rlwe.SecretKey, pk *rlwe.PublicKey, rlk *rlwe.RelinearizationKey,
	encoder ckks.Encoder, encryptor ckks.Encryptor,
) *CryptoParams {

	encoders := make(chan ckks.Encoder, 1)
	encoders <- encoder

	encryptors := make(chan ckks.Encryptor, 1)
	encryptors <- encryptor

	return &CryptoParams{
		Params: params,
		Sk:     sk,
		Rlk:    rlk,
		Pk:     pk,
		// Kgen:       kgen,
		encoders:   encoders,
		encryptors: encryptors,
	}
}

// NewCryptoParams initializes CryptoParams with the given values
func NewCryptoParams(params ckks.Parameters, sk, aggregateSk *rlwe.SecretKey, pk *rlwe.PublicKey, rlk *rlwe.RelinearizationKey) *CryptoParams {
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
		// Kgen:        kgen,

		encoders:   encoders,
		encryptors: encryptors,
		decryptors: decryptors,
		evaluators: evaluators,
	}
}

// Kgen return a new key generator
func (cp *CryptoParams) Kgen() ckks.KeyGenerator {
	return ckks.NewKeyGenerator(cp.Params)
}

// MarshalBinary marshall the sk, pk
func (cp *CryptoParams) MarshalBinary() ([]byte, []byte) {
	var dsk, dpk []byte
	var err error
	if dsk, err = cp.Sk.MarshalBinary(); err != nil {
		panic("fail in sk marshal")
	}
	if dpk, err = cp.Pk.MarshalBinary(); err != nil {
		panic("fail in pk marshal")
	}
	if len(dsk) == 0 {
		panic("fail in loading marshal bytes")
	}
	return dsk, dpk
}

// RetrieveCommonParams retrieve sk, pj
func (cp *CryptoParams) RetrieveCommonParams(dsk, dpk []byte) {
	sk := new(rlwe.SecretKey)
	if err := sk.UnmarshalBinary(dsk); err != nil {
		panic("fail to unmarshall sk")
	}
	pk := new(rlwe.PublicKey)
	if err := pk.UnmarshalBinary(dpk); err != nil {
		panic("fail to unmarshall sk")
	}
	cp.Sk = sk
	cp.Pk = pk
	cp.Rlk = cp.Kgen().GenRelinearizationKey(sk)

	close(cp.encryptors)
	cp.encryptors = make(chan ckks.Encryptor, ThreadsCount)
	for i := 0; i < ThreadsCount; i++ {
		cp.encryptors <- ckks.NewEncryptorFromPk(cp.Params, pk)
	}
	close(cp.decryptors)
	cp.decryptors = make(chan ckks.Decryptor, ThreadsCount)
	for i := 0; i < ThreadsCount; i++ {
		cp.decryptors <- ckks.NewDecryptor(cp.Params, sk)
	}
}

// SetDecryptors sets the decryptors in the CryptoParams object
func (cp *CryptoParams) SetDecryptors(params ckks.Parameters, sk *rlwe.SecretKey) {
	decryptors := make(chan ckks.Decryptor, ThreadsCount)
	for i := 0; i < ThreadsCount; i++ {
		decryptors <- ckks.NewDecryptor(params, sk)
	}
	cp.decryptors = decryptors
}

// SetEncryptors sets the encryptors in the CryptoParams object
func (cp *CryptoParams) SetEncryptors(params ckks.Parameters, pk *rlwe.PublicKey) {
	encryptors := make(chan ckks.Encryptor, ThreadsCount)
	for i := 0; i < ThreadsCount; i++ {
		encryptors <- ckks.NewEncryptorFromPk(params, pk)
	}
	cp.encryptors = encryptors
}

// SetEvaluator sets the encryptors in the CryptoParams object
func (cp *CryptoParams) SetEvaluator(eval ckks.Evaluator) {
	evaluators := make(chan ckks.Evaluator, ThreadsCount)
	for i := 0; i < ThreadsCount; i++ {
		evaluators <- eval.ShallowCopy()
	}
}

// NewCryptoParamsForNetwork initializes a set of nbrNodes CryptoParams each containing: keys, encoder, encryptor, decryptor, etc.
func NewCryptoParamsForNetwork(params ckks.Parameters, nbrNodes int) []*CryptoParams {
	kgen := ckks.NewKeyGenerator(params)

	aggregateSk := ckks.NewSecretKey(params)
	skList := make([]*rlwe.SecretKey, nbrNodes)
	rq, _ := ring.NewRing(params.N(), append(params.Q(), params.P()...))

	for i := 0; i < nbrNodes; i++ {
		skList[i] = kgen.GenSecretKey()
		rq.Add(aggregateSk.Value, skList[i].Value, aggregateSk.Value)
	}
	pk := kgen.GenPublicKey(aggregateSk)

	ret := make([]*CryptoParams, nbrNodes)
	for i := range ret {
		rlk := kgen.GenRelinearizationKey(aggregateSk)
		ret[i] = NewCryptoParams(params, skList[i], aggregateSk, pk, rlk)
	}
	return ret
}

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
	ringQP, err := ring.NewRing(cryptoParams.Params.N(), append(cryptoParams.Params.Q(), cryptoParams.Params.P()...))
	if err != nil {
		return nil, fmt.Errorf("creating new ring: %v", err)
	}

	sampler := ring.NewUniformSampler(prng, ringQP)
	return sampler, nil
}

// GetScaleByLevel returns a scale for a given level
func (cp *CryptoParams) GetScaleByLevel(level uint64) float64 {
	return float64(cp.Params.Q()[level])
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

// GetEncoder for debug use, return the encoder
func (cp *CryptoParams) GetEncoder() ckks.Encoder {
	return ckks.NewEncoder(cp.Params)
}

// GetEncryptor for debug use, return encryptor
func (cp *CryptoParams) GetEncryptor() ckks.Encryptor {
	if cp.AggregateSk != nil {
		return ckks.NewEncryptorFromSk(cp.Params, cp.AggregateSk)
	}
	return ckks.NewEncryptorFromSk(cp.Params, cp.Sk)
}

// GetDecryptor for debug use, return encryptor
func (cp *CryptoParams) GetDecryptor() ckks.Decryptor {
	tmp := <-cp.decryptors
	cp.decryptors <- tmp
	return tmp
}

// GetEvaluator for debug use, return a evaluator
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
