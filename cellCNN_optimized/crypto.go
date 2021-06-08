package cellCNN

import (
	"bytes"
	"encoding"
	"encoding/gob"
	"errors"
	"fmt"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"go.dedis.ch/onet/v3/log"
	"io/ioutil"
	"os"
	"path/filepath"
)

// CryptoParams aggregates all ckks scheme information
type CryptoParams struct {
	Sk          *ckks.SecretKey
	AggregateSk *ckks.SecretKey
	Pk          *ckks.PublicKey
	Rlk         *ckks.RelinearizationKey
	RotKs       *ckks.RotationKeySet
	Params      *ckks.Parameters

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
	rlk         *ckks.RelinearizationKey
	rotKs       *ckks.RotationKeySet
}

type cryptoParamsMarshalable struct {
	Params      *ckks.Parameters
	Sk          []*ckks.SecretKey
	AggregateSk *ckks.SecretKey
	Pk          *ckks.PublicKey
	Rlk         *ckks.RelinearizationKey
	RotKs       *ckks.RotationKeySet
}

var _ encoding.BinaryMarshaler = new(CryptoParamsForNetwork)
var _ encoding.BinaryUnmarshaler = new(CryptoParamsForNetwork)

// NewCryptoParams initializes CryptoParams with the given values
func NewCryptoParams(params *ckks.Parameters, sk, aggregateSk *ckks.SecretKey, pk *ckks.PublicKey, rlk *ckks.RelinearizationKey) *CryptoParams {
	evaluators := make(chan ckks.Evaluator, ThreadsCount)
	for i := 0; i < ThreadsCount; i++ {
		evaluators <- ckks.NewEvaluator(params, ckks.EvaluationKey{
			Rlk:  rlk,
			Rtks: nil,
		})
	}

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
		decryptors <- ckks.NewDecryptor(params, aggregateSk)
	}

	return &CryptoParams{
		Params:      params,
		Sk:          sk,
		AggregateSk: aggregateSk,
		Pk:          pk,
		Rlk:         rlk,

		encoders:   encoders,
		encryptors: encryptors,
		decryptors: decryptors,
		evaluators: evaluators,
	}
}

// NewCryptoParamsForNetwork initializes a set of nbrNodes CryptoParams each containing: keys, encoder, encryptor, decryptor, etc.
func NewCryptoParamsForNetwork(params *ckks.Parameters, nbrNodes int) []*CryptoParams {
	kgen := ckks.NewKeyGenerator(params)

	aggregateSk := ckks.NewSecretKey(params)
	skList := make([]*ckks.SecretKey, nbrNodes)
	rq, _ := ring.NewRing(params.N(), append(params.Qi(), params.Pi()...))

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

// NewCryptoParamsFromPath reads given path and return the parsed CryptoParams
func NewCryptoParamsFromPath(path string) ([]*CryptoParams, error) {
	encoded, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	ret := new(CryptoParamsForNetwork)
	if err := ret.UnmarshalBinary(encoded); err != nil {
		return nil, fmt.Errorf("decode: %v", err)
	}

	cryptoParamsList := make([]*CryptoParams, len(ret.sk))
	for i := range cryptoParamsList {
		cryptoParamsList[i] = NewCryptoParams(ret.params, ret.sk[i], ret.aggregateSk, ret.pk, ret.rlk)
		cryptoParamsList[i].RotKs = ret.rotKs
	}

	return cryptoParamsList, nil
}

// ReadOrGenerateCryptoParams reads (if files already exist) or creates the crypto information and stores it in files for future use
func ReadOrGenerateCryptoParams(hosts int, defaultN *ckks.Parameters, rootPath string, generateRotKeys bool) []*CryptoParams {
	var cryptoParamsAreNew bool
	cryptoParamsList, err := NewCryptoParamsFromPath(rootPath)
	if errors.Is(err, os.ErrNotExist) {
		log.Warn("inexisting cache, creating a new crypto params")
		cryptoParamsList = NewCryptoParamsForNetwork(defaultN, hosts)
		cryptoParamsAreNew = true
	} else if err != nil {
		panic(err)
	} else if len(cryptoParamsList) != hosts {
		log.Warn("invalid cache, creating a new crypto params")
		cryptoParamsList = NewCryptoParamsForNetwork(defaultN, hosts)
		cryptoParamsAreNew = true
	}

	if generateRotKeys || cryptoParamsList[0].RotKs == nil {
		log.Warn("generating new rotation keys")
		rP.Slots = cryptoParamsList[0].GetSlots()
		rotations, err := GetRotationsForMLType(mltype, rP, nnP)
		if err != nil {
			panic(fmt.Errorf("generate rotation keys: %v", err))
		}
		cryptoParamsList[0].SetRotKeys(rotations)

		secretKeysList := make([]*ckks.SecretKey, len(cryptoParamsList))
		for i, cp := range cryptoParamsList {
			secretKeysList[i] = cp.Sk
			cryptoParamsList[i].RotKs = cryptoParamsList[0].RotKs
		}

		if cryptoParamsAreNew {
			log.Warnf("writing new crypto params to %v", rootPath)
			if err := cryptoParamsList[0].WriteParamsToFile(rootPath, secretKeysList); err != nil {
				panic(err)
			}
		}
	}

	return cryptoParamsList
}

// MarshalBinary encode into a []byte accepted by UnmarshalBinary
func (cp *CryptoParamsForNetwork) MarshalBinary() ([]byte, error) {
	var ret bytes.Buffer
	encoder := gob.NewEncoder(&ret)

	err := encoder.Encode(cryptoParamsMarshalable{
		Params:      cp.params,
		Sk:          cp.sk,
		AggregateSk: cp.aggregateSk,
		Pk:          cp.pk,
		Rlk:         cp.rlk,
		RotKs:       cp.rotKs,
	})
	if err != nil {
		return nil, fmt.Errorf("encode minimal crypto params: %v", err)
	}

	return ret.Bytes(), nil
}

// UnmarshalBinary decode a []byte created by MarshalBinary
func (cp *CryptoParamsForNetwork) UnmarshalBinary(data []byte) error {
	decoder := gob.NewDecoder(bytes.NewBuffer(data))

	decodeParams := new(cryptoParamsMarshalable)
	if err := decoder.Decode(decodeParams); err != nil {
		return fmt.Errorf("decode minimal crypto params: %v", err)
	}

	cp.params = decodeParams.Params
	cp.sk = decodeParams.Sk
	cp.aggregateSk = decodeParams.AggregateSk
	cp.pk = decodeParams.Pk
	cp.rlk = decodeParams.Rlk
	cp.rotKs = decodeParams.RotKs

	return nil
}

// WriteParamsToFile writes the crypto params to a toml file including the keys' filenames
func (cp *CryptoParams) WriteParamsToFile(path string, secretKeysList []*ckks.SecretKey) error {
	if err := os.MkdirAll(filepath.Dir(path), os.ModePerm); err != nil {
		return fmt.Errorf("creating parents dirs of %v: %w", path, err)
	}

	cpSave := &CryptoParamsForNetwork{
		sk:          secretKeysList,
		aggregateSk: cp.AggregateSk,
		pk:          cp.Pk,
		rlk:         cp.Rlk,
		rotKs:       cp.RotKs,
		params:      cp.Params,
	}

	encoded, err := cpSave.MarshalBinary()
	if err != nil {
		return fmt.Errorf("encode: %v", err)
	}

	if err := ioutil.WriteFile(path, encoded, 0666); err != nil {
		return fmt.Errorf("write file: %v", err)
	}

	return nil
}
