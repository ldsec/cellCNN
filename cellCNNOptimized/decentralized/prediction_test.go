package decentralized_test

import(
	"fmt"
	"testing"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/cellCNN/cellCNNOptimized"
)

func BenchmarkPrediction(b *testing.B){
	params := cellCNN.GenParams()

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	pt := ckks.NewPlaintext(params, 4, params.Scale())
	encryptor := ckks.NewEncryptorFromSk(params, sk)
	encoder := ckks.NewEncoder(params)

	rotations := cellCNN.GetRotationKeysIndex(params)

	rotKeys := kgen.GenRotationKeysForRotations(rotations, true, sk)
	rlk := kgen.GenRelinearizationKey(sk)

	eval := ckks.NewEvaluator(params, rlwe.EvaluationKey{rlk, rotKeys})

	CtW := encryptor.EncryptNew(pt)
	CtC := encryptor.EncryptNew(pt)

	batchSize := params.Slots() / (cellCNN.Filters * cellCNN.Classes)

	XBatch := cellCNN.GenRandomRealMatrices(cellCNN.Cells, cellCNN.Features, batchSize)

	ciphertexts := make([]*ckks.Ciphertext, cellCNN.Features>>1)
	for i := range ciphertexts {
		ciphertexts[i] = ckks.NewCiphertext(params, 1, 4, params.Scale())
	}

	b.Run(fmt.Sprintf("Prediction/Client/Ciphers=%d/batchSize=%d/", len(ciphertexts), batchSize), func(b *testing.B){
		for i := 0; i < b.N; i++{
			cellCNN.EncryptForPrediction(XBatch, encoder, encryptor, params, ciphertexts)
		}
	})

	b.Run(fmt.Sprintf("Prediction/Server/batchSize=%d/", batchSize), func(b *testing.B){
		cellCNN.EncryptForPrediction(XBatch, encoder, encryptor, params, ciphertexts)
		b.ResetTimer()
		for i := 0; i < b.N; i++{
			cellCNN.Predict(ciphertexts, CtC, CtW, params, eval)
		}

	})
}