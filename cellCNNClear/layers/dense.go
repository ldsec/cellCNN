package layers

import (
	"github.com/ldsec/cellCNN/cellCNN_clear/utils"
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
	//coeffsSigmoid := []float64{0.5, 0.6997, 0, -0.2649} //coeffs for interval 3
	//	coeffsSigmoid :=[]float64{0.5, 0.9917, 0, -0.5592} //coeffs for interval 5

	coeffsSigmoid := []float64{0.5, 1.2010, 0, -0.8156} //coeffs for interval 8
	if dense.weights == nil || dense.last_input == nil {
		dense.firstMoment = false
		dense.vt = mat.NewDense(nfilters, dense.Nclasses, nil)
		//c :=[]float64{0.270457, -0.004343, 0.290110, 0.172072, -0.184941, -0.075417, 0.334569, 0.362849, -0.001117, -0.172576, 0.388567, -0.038715}
		c := utils.WeightsInit(nfilters*dense.Nclasses, float64(nfilters))
		dense.weights = mat.NewDense(nfilters, dense.Nclasses, c)
		dense.last_input = mat.NewDense(nsamples, nfilters, nil)
		dense.u = mat.NewDense(nsamples, dense.Nclasses, nil)
		dense.activation = func(z float64) float64 {
			if dense.ApproxInterval > 0 {

				return utils.SigmoidApproxClear(z, dense.ApproxInterval, coeffsSigmoid)
			} else {
				return utils.Sigmoid(z)
			}
		}
		dense.d_activation = func(z float64) float64 {
			if dense.ApproxInterval > 0 {
				d, _ := utils.SigmoidDApproxClear(z, dense.ApproxInterval, coeffsSigmoid)
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

	//fmt.Printf(" %v\n", mat.Formatted(dense.weights, mat.Prefix(" "), mat.Excerpt(3)))

	return next_error, scaledDW
}

func (dense *Dense_n) GetWeights() *mat.Dense {
	return dense.weights
}
