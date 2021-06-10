package utils

import (
	"math/rand"

	"github.com/ldsec/cellCNN/cellCNN_clear/protocols/common"
	"github.com/ldsec/lattigo/v2/ckks"
	"gonum.org/v1/gonum/mat"
)

func GetRandomBatch(
	dataset *common.CnnDataset, batchSize int, params ckks.Parameters, encoder ckks.Encoder,
	sts *CellCnnSettings,
) ([]*ckks.Plaintext, []*mat.Dense, []float64) {
	// X := dataset.X
	// y := dataset.Y

	// // make a new batch
	// newBatch := make([]*mat.Dense, batchSize)
	// newBatchLabels := make([]float64, batchSize)
	// for j := 0; j < len(newBatch); j++ {
	// 	randi := rand.Intn(len(X))
	// 	newBatch[j] = X[randi]
	// 	newBatchLabels[j] = y[randi]
	// }
	newBatch := make([]*mat.Dense, batchSize)
	newBatchLabels := make([]float64, batchSize)

	for i := 0; i < batchSize; i++ {
		newBatch[i] = GenRandomMatrix(sts.Ncells, sts.Nmakers)
		newBatchLabels[i] = float64(rand.Intn(sts.Nclasses))
	}

	plaintextSlice := Batch2PlainSlice(newBatch, params, encoder)
	return plaintextSlice, newBatch, newBatchLabels
}
