package centralized

import (
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
)

// BatchProcessing conduct batch forward and backward.
// you can set isMomentum=false to compute only the scaled gradients in decentralized version
// return the predictions as a slice of ciphertext
func (c *CellCNN) BatchProcessing(inputBatch []*ckks.Plaintext, labels []float64, isMomentum bool) ([]*ckks.Ciphertext, float64, float64) {

	preds := make([]*ckks.Ciphertext, len(inputBatch))
	var gradAccumulator *Gradients = nil
	suppressGradientModify := 0.0

	tForward := 0.0
	tBackward := 0.0

	for i, input := range inputBatch {
		// forward one
		tFwdStart := time.Now()
		out1 := c.conv1d.Forward(input, nil, c.cnnSettings, c.evaluator, c.params)
		out2 := c.dense.Forward(out1, nil, c.cnnSettings, c.evaluator, c.encoder, c.params)
		tFwdOne := time.Since(tFwdStart).Seconds()

		tBcwStart := time.Now()
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
			gradAccumulator.Aggregate(append(pgConv, pgDense), c.evaluator)
		}
		tBcwOne := time.Since(tBcwStart).Seconds()

		tForward += tFwdOne
		tBackward += tBcwOne
	}

	tScaleStart := time.Now()
	// compute the modified gradients
	c.conv1d.ComputeScaledGradient(gradAccumulator.filters, c.cnnSettings, c.params, c.evaluator, c.encoder, c.lr)
	c.dense.ComputeScaledGradient(gradAccumulator.dense, c.cnnSettings, c.params, c.evaluator, c.encoder, c.lr)

	// with momentum
	if isMomentum {
		c.conv1d.ComputeScaledGradientWithMomentum(c.conv1d.GetGradient(), c.cnnSettings, c.params, c.evaluator, c.encoder, c.momentum)
		c.dense.ComputeScaledGradientWithMomentum(c.dense.GetGradient(), c.cnnSettings, c.params, c.evaluator, c.encoder, c.momentum)
	}

	tScaleOne := time.Since(tScaleStart).Seconds()

	tBackward += tScaleOne

	return preds, tForward, tBackward
}
