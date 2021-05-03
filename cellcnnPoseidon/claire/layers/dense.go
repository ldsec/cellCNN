package layers

import (
	"github.com/ldsec/cellCNN/celcnnPoseidon/approx"
	"github.com/ldsec/cellCNN/celcnnPoseidon/claire/utils"
	"github.com/ldsec/cellCNN/internal/leastsquares"
	"gonum.org/v1/gonum/mat"
)

// dense output layer for classification (nclasses > 1)
type Dense_n struct {
	Nclasses       int
	weights        *mat.Dense // nfilters x nclasses
	last_input     *mat.Dense // nsamples x nfilters
	u              *mat.Dense
	ApproxInterval float64
	activation     func(float64) float64
	d_activation   func(float64) float64
	vt             *mat.Dense // nfilters x nclasses
	firstMoment    bool
}

// Forward pass of the Dense_n layer, using newWeights if not nil
// if ApproxInterval > 0, activation is approximated sigmoid in given interval
// else activation is sigmoid
func (dense *Dense_n) Forward(input *mat.Dense, newWeights *mat.Dense) *mat.Dense {

	nsamples, nfilters := input.Dims()

	if dense.weights == nil || dense.last_input == nil {
		dense.firstMoment = false
		dense.vt = mat.NewDense(nfilters, dense.Nclasses, nil)
		dense.weights = mat.NewDense(nfilters, dense.Nclasses, utils.WeightsInit(nfilters*dense.Nclasses, float64(nfilters)))
		dense.last_input = mat.NewDense(nsamples, nfilters, nil)
		dense.u = mat.NewDense(nsamples, dense.Nclasses, nil)
		dense.activation = func(z float64) float64 {
			if dense.ApproxInterval > 0 {
				coeffsSigmoid, _ := leastsquares.GetCoefficients(3, dense.ApproxInterval)
				return approx.SigmoidApproxClear(z, dense.ApproxInterval, coeffsSigmoid)
			} else {
				return utils.Sigmoid(z)
			}
		}
		dense.d_activation = func(z float64) float64 {
			if dense.ApproxInterval > 0 {
				d, _ := approx.SigmoidDApproxClear(z, 3, dense.ApproxInterval)
				return d
			} else {
				return utils.Sigmoid(z) * (1 - utils.Sigmoid(z))
			}
		}

	}

	if newWeights != nil {
		dense.weights = newWeights
	}

	dense.last_input.Copy(input)
	dense.u.Mul(input, dense.weights)

	output := mat.NewDense(nsamples, dense.Nclasses, nil)
	output.Apply(utils.ToApply(dense.activation), dense.u)

	return output
}

// Backward performs backpropagation of the Dense_n layer given the loss/error
// returns the next error and the change dW scaled by the learn_rate
func (dense *Dense_n) Backward(error *mat.Dense, learn_rate, momentum float64) (*mat.Dense, *mat.Dense) {
	r, c := dense.weights.Dims()

	dense.u.Apply(utils.ToApply(dense.d_activation), dense.u)
	error.MulElem(dense.u, error)

	dW := mat.NewDense(r, c, nil)
	dW.Mul(dense.last_input.T(), error)
	scaledDW := mat.NewDense(r, c, nil)
	scaledDW.Scale(learn_rate, dW)

	r_in, c_in := dense.last_input.Dims()
	next_error := mat.NewDense(r_in, c_in, nil)
	next_error.Mul(error, dense.weights.T())

	if momentum > 0 {
		if dense.firstMoment == false {
			dense.vt = scaledDW
			dense.firstMoment = true
		} else {
			dense.vt.Scale(momentum, dense.vt)
			dense.vt.Add(dense.vt, scaledDW)
		}
		dense.weights.Sub(dense.weights, dense.vt)
	} else {
		dense.weights.Sub(dense.weights, scaledDW)
	}
	return next_error, scaledDW
}

func (dense *Dense_n) GetWeights() *mat.Dense {
	return dense.weights
}
