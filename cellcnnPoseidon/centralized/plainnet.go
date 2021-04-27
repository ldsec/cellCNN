package centralized

import (
	"github.com/ldsec/cellCNN/cellCNN_clear/layers"
	"gonum.org/v1/gonum/mat"
)

type PlainNet struct {
	ncells   int
	nmakers  int
	nfilters int
	nclasses int
	conv     *layers.Conv1D
	pool     *layers.Pool
	dense    *layers.Dense_n
}

func (p *PlainNet) ForwardBatch(input []*mat.Dense, cw, dw *mat.Dense) *mat.Dense {
	out1 := p.conv.Forward(input, cw)
	out2 := p.pool.Forward(out1)
	out2 = p.dense.Forward(out2, dw)
	return out2
}

func (p *PlainNet) Backward(gradient *mat.Dense, learn_rate float64, momentum float64) (*mat.Dense, *mat.Dense) {
	delta2, dDense := p.dense.Backward(gradient, learn_rate, momentum)
	delta1 := p.pool.Backward(delta2)
	dConv := p.conv.Backward(delta1, learn_rate, momentum)
	return dDense, dConv
}
