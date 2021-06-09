package centralized

import (
	"github.com/ldsec/cellCNN/cellcnnPoseidon/utils"
	"github.com/ldsec/lattigo/v2/ckks"
)

// ForwardBatch conduct batch forward.
// return the preds of each input, modified grad (with momentum and lr) called by GetGradient
func (c *CellCNN) BatchProcessing(inputBatch []*ckks.Plaintext, labels []float64) []*ckks.Ciphertext {

	LeftMostMask := utils.GenSliceWithOneAt(c.params.Slots(), []int{0})
	poolMask := c.encoder.EncodeNTTAtLvlNew(c.params.MaxLevel(), LeftMostMask, c.params.LogSlots())
	preds := make([]*ckks.Ciphertext, len(inputBatch))
	var gradAccumulator *Gradients = nil
	suppressGradientModify := 0.0

	for i, input := range inputBatch {
		// forward one
		out1 := c.conv1d.Forward(input, nil, c.cnnSettings, c.evaluator, c.params, poolMask)
		out2 := c.dense.Forward(out1, nil, c.cnnSettings, c.evaluator, c.encoder, c.params)
		// record the preds
		preds[i] = out2.CopyNew()

		// backward one
		err0 := c.ComputeLossOne(out2, labels[i])
		// record the pure gradient
		dsErr, pgDense := c.dense.Backward(err0, c.cnnSettings, c.params, c.evaluator, c.encoder, c.sk, suppressGradientModify)
		pgConv := c.conv1d.Backward(dsErr, c.cnnSettings, c.params, c.evaluator, c.encoder, suppressGradientModify)

		// agg batch gradients
		if i == 0 {
			gradAccumulator = &Gradients{pgConv, pgDense}
		} else {
			gradAccumulator.AggregateCt(pgConv, pgDense, c.evaluator)
		}
	}

	// compute the modified gradients
	c.conv1d.ComputeScaledGradient(gradAccumulator.filters, c.cnnSettings, c.params, c.evaluator, c.encoder, c.lr)
	c.dense.ComputeScaledGradient(gradAccumulator.dense, c.cnnSettings, c.params, c.evaluator, c.encoder, c.lr)

	// with momentum
	if c.momentum != 0 {
		c.conv1d.ComputeScaledGradientWithMomentum(c.conv1d.GetGradient(), c.cnnSettings, c.params, c.evaluator, c.encoder)
		c.dense.ComputeScaledGradientWithMomentum(c.dense.GetGradient(), c.cnnSettings, c.params, c.evaluator, c.encoder)
	}

	return preds
}
