package centralized

import (
	"math/rand"
	"time"

	"github.com/ldsec/cellCNN/celcnnPoseidon/layers"
	"github.com/ldsec/cellCNN/celcnnPoseidon/utils"
	"github.com/ldsec/lattigo/v2/ckks"
	"gonum.org/v1/gonum/mat"
)

func MakeRandomBatch(sts *layers.CellCnnSettings, batchsize int) []*mat.Dense {
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

func ForwardAndBackwardNiter(c *CellCNN, niter int, batchSize int) {
	// preparing public values
	nfilters := c.cnnSettings.Nfilters
	nclasses := c.cnnSettings.Nclasses

	// initialize masks required
	// conv1d left most mask
	LeftMostMask := make([]complex128, c.params.Slots())
	LeftMostMask[0] = complex(float64(1), 0)
	poolMask := c.encoder.EncodeNTTAtLvlNew(c.params.MaxLevel(), LeftMostMask, c.params.LogSlots())

	// dense maskMap to collect all results into one ciphertext
	maskMap := make(map[int]*ckks.Plaintext)
	for i := 0; i < nclasses; i++ {
		maskMap[i*(nfilters-1)] = func() *ckks.Plaintext {
			tmpMask := make([]complex128, c.params.Slots())
			tmpMask[i] = complex(float64(1), 0)
			return c.encoder.EncodeNTTAtLvlNew(c.params.MaxLevel(), tmpMask, c.params.LogSlots())
		}()
	}

	glb_tf := make([]float64, 3)
	glb_tb := make([]float64, 3)

	// start training
	for i := 1; i <= niter; i++ {
		// make a new batch
		batch := MakeRandomBatch(c.cnnSettings, batchSize)
		plaintextSlice := c.Batch2PlainSlice(batch)

		avg_tf := make([]float64, 3)
		avg_tb := make([]float64, 3)
		for j := 0; j < batchSize; j++ {
			tf, tb := c.ForwardAndBackwardOne(plaintextSlice[j], nil, nil, poolMask, maskMap)

			for i := 0; i < len(avg_tf); i++ {
				avg_tf[i] += tf[i]
				avg_tb[i] += tb[i]
			}
			utils.PrintTime(tf, &j, "Forward One")
			utils.PrintTime(tb, &j, "Backward One")
		}
		for k := 0; k < len(avg_tf); k++ {
			avg_tf[k] /= float64(batchSize)
			avg_tb[k] /= float64(batchSize)
		}
		utils.PrintTime(avg_tf, &i, "\nAVG Forward One in Batch")
		utils.PrintTime(avg_tb, &i, "\nAVG Backward One in Batch")

		for q := 0; q < len(glb_tf); q++ {
			glb_tf[q] += avg_tf[q]
			glb_tb[q] += avg_tb[q]
		}
	}
	for q := 0; q < len(glb_tf); q++ {
		glb_tf[q] /= float64(niter)
		glb_tb[q] /= float64(niter)
	}

	utils.PrintTime(glb_tf, &niter, "\nAVG Forward One in All Batches")
	utils.PrintTime(glb_tb, &niter, "\nAVG Backward One in All Batches")
}
