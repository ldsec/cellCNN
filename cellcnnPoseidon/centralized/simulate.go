package centralized

import (
	"math/rand"
	"time"

	"github.com/ldsec/cellCNN/cellcnnPoseidon/utils"
	"gonum.org/v1/gonum/mat"
)

// MakeRandomBatch for debug use only, make a random batch of data in [0.0, 1.0)
func MakeRandomBatch(sts *utils.CellCnnSettings, batchsize int) []*mat.Dense {
	res := make([]*mat.Dense, batchsize)
	nm := sts.Nmakers
	nc := sts.Ncells
	generator := rand.New(rand.NewSource(time.Now().UnixNano()))

	for i := range res {
		backup := make([]float64, nm*nc)
		for j := 0; j < nm*nc; j++ {
			backup[j] = generator.Float64()
		}
		res[i] = mat.NewDense(nc, nm, backup)
	}
	return res
}

// ForwardAndBackwardNiter for debug use only, test the forward and backward time on randomly generated data.
func ForwardAndBackwardNiter(c *CellCNN, niter int, batchSize int) {

	glbTf := make([]float64, 3)
	glbTb := make([]float64, 3)

	// start training
	for i := 1; i <= niter; i++ {
		// make a new batch
		batch := MakeRandomBatch(c.cnnSettings, batchSize)
		plaintextSlice := c.Batch2PlainSlice(batch)

		avgTf := make([]float64, 3)
		avgTb := make([]float64, 3)
		for j := 0; j < batchSize; j++ {
			tf, tb := c.ForwardAndBackwardOne(plaintextSlice[j], nil, nil)

			for i := 0; i < len(avgTf); i++ {
				avgTf[i] += tf[i]
				avgTb[i] += tb[i]
			}
			utils.PrintTime(tf, &j, "Forward One")
			utils.PrintTime(tb, &j, "Backward One")
		}
		for k := 0; k < len(avgTf); k++ {
			avgTf[k] /= float64(batchSize)
			avgTb[k] /= float64(batchSize)
		}
		utils.PrintTime(avgTf, &i, "\nAVG Forward One in Batch")
		utils.PrintTime(avgTb, &i, "\nAVG Backward One in Batch")

		for q := 0; q < len(glbTf); q++ {
			glbTf[q] += avgTf[q]
			glbTb[q] += avgTb[q]
		}
	}
	for q := 0; q < len(glbTf); q++ {
		glbTf[q] /= float64(niter)
		glbTb[q] /= float64(niter)
	}

	utils.PrintTime(glbTf, &niter, "\nAVG Forward One in All Batches")
	utils.PrintTime(glbTb, &niter, "\nAVG Backward One in All Batches")
}
