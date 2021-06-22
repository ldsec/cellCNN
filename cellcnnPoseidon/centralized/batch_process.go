package centralized

import (
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
)

// ForwardBatch conduct batch forward.
// return the preds of each input and the sum forward/backward time in s
// modified grad (with momentum and lr) called by GetGradient
func (c *CellCNN) BatchProcessing(inputBatch []*ckks.Plaintext, labels []float64, isMomentum bool) ([]*ckks.Ciphertext, float64, float64) {

	preds := make([]*ckks.Ciphertext, len(inputBatch))
	var gradAccumulator *Gradients = nil
	suppressGradientModify := 0.0

	t_forward := 0.0
	t_backward := 0.0

	for i, input := range inputBatch {
		// forward one
		t_fwd_start := time.Now()
		out1 := c.conv1d.Forward(input, nil, c.cnnSettings, c.evaluator, c.params)
		out2 := c.dense.Forward(out1, nil, c.cnnSettings, c.evaluator, c.encoder, c.params)
		t_fwd_one := time.Since(t_fwd_start).Seconds()

		t_bcw_start := time.Now()
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
		t_bcw_one := time.Since(t_bcw_start).Seconds()

		t_forward += t_fwd_one
		t_backward += t_bcw_one
	}

	t_scale_start := time.Now()
	// compute the modified gradients
	c.conv1d.ComputeScaledGradient(gradAccumulator.filters, c.cnnSettings, c.params, c.evaluator, c.encoder, c.lr)
	c.dense.ComputeScaledGradient(gradAccumulator.dense, c.cnnSettings, c.params, c.evaluator, c.encoder, c.lr)

	// with momentum
	if isMomentum {
		c.conv1d.ComputeScaledGradientWithMomentum(c.conv1d.GetGradient(), c.cnnSettings, c.params, c.evaluator, c.encoder, c.momentum)
		c.dense.ComputeScaledGradientWithMomentum(c.dense.GetGradient(), c.cnnSettings, c.params, c.evaluator, c.encoder, c.momentum)
	}

	t_scale_one := time.Since(t_scale_start).Seconds()

	t_backward += t_scale_one

	return preds, t_forward, t_backward
}
