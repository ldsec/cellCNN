package centralized

import (
	"github.com/ldsec/cellCNN/cellCNNClear/layers"
	"gonum.org/v1/gonum/mat"
)

// PlainNet is a holder of cellCNNClear
type PlainNet struct {
	ncells   int
	nmakers  int
	nfilters int
	nclasses int
	conv     *layers.Conv1D
	pool     *layers.Pool
	dense    *layers.Dense_n
}

// ForwardBatch forward a batch of samples, return the prediction as a matrix
func (p *PlainNet) ForwardBatch(input []*mat.Dense, cw, dw *mat.Dense) *mat.Dense {
	out1 := p.conv.Forward(input, cw)
	out2 := p.pool.Forward(out1)
	out2 = p.dense.Forward(out2, dw)
	return out2
}

// Backward return the gradient for conv and dense
func (p *PlainNet) Backward(gradient *mat.Dense, lr float64, momentum float64) (*mat.Dense, *mat.Dense) {
	delta2, dDense := p.dense.Backward(gradient, lr, momentum)
	delta1 := p.pool.Backward(delta2)
	dConv := p.conv.Backward(delta1, lr, momentum)
	return dConv, dDense
}
