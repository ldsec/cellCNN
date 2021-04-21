package layers

import (
	"github.com/ldsec/cellCNN/cellCNN_clear/utils"
	"gonum.org/v1/gonum/mat"
)

// a 1D convolutional layer
type Conv1D struct {
	Nfilters     int
	filters      *mat.Dense   // nmarkers x nfilters
	last_input   []*mat.Dense // nsamples x ncells x nmarkers
	activation   func(float64) float64
	d_activation func(float64) float64
	u            []*mat.Dense
	vt           *mat.Dense // nmarkers x nfilters
	firstMoment  bool
}

// Forward computes a forward pass of the Conv1D layer, using newFilters if not nil
// activation function is identity
func (conv *Conv1D) Forward(input []*mat.Dense, newFilters *mat.Dense) []*mat.Dense {
	ncells, nmarkers := input[0].Dims()

	if conv.last_input == nil {
		conv.last_input = make([]*mat.Dense, len(input))
	}
	copy(conv.last_input, input) // remember last input

	// initialize filters and activation
	if conv.filters == nil {
		conv.filters = mat.NewDense(nmarkers, conv.Nfilters, utils.WeightsInit(nmarkers*conv.Nfilters, float64(nmarkers)))
		conv.u = make([]*mat.Dense, len(input))
		conv.activation = func(z float64) float64 {
			return z
		}
		conv.d_activation = func(z float64) float64 {
			return 1
		}
		conv.firstMoment = false
		conv.vt = mat.NewDense(nmarkers, conv.Nfilters, nil)
	}

	// replace filters
	if newFilters != nil {
		conv.filters = newFilters
	}

	output := make([]*mat.Dense, len(input))
	for i := range input {
		output[i] = mat.NewDense(ncells, conv.Nfilters, nil)
		conv.u[i] = mat.NewDense(ncells, conv.Nfilters, nil)
		conv.u[i].Mul(input[i], conv.filters)
		output[i].Apply(utils.ToApply(conv.activation), conv.u[i])
	}

	return output
}

// Backward computes the backpropagation of the Conv1D layer, given error/loss
// returns the change dW scaled by the learn_rate
func (conv *Conv1D) Backward(error []*mat.Dense, learn_rate, momentum float64) *mat.Dense {
	nmarkers, nfilters := conv.filters.Dims()
	dW := mat.NewDense(nmarkers, nfilters, nil)
	temp := mat.NewDense(nmarkers, nfilters, nil)

	for i := range error {
		//fmt.Println(conv.u[i])
		conv.u[i].Apply(utils.ToApply(conv.d_activation), conv.u[i])
		//fmt.Println(conv.u[i])
		//os.Exit(0)
		error[i].MulElem(conv.u[i], error[i])
		temp.Mul(conv.last_input[i].T(), error[i])
		dW.Add(dW, temp)
	}

	scaledDW := mat.NewDense(nmarkers, nfilters, nil)
	//fmt.Print(learn_rate/float64(len(conv.u)))
	scaledDW.Scale(learn_rate, dW)

	if momentum > 0 {
		if conv.firstMoment == false {
			conv.vt = scaledDW
			conv.firstMoment = true
		} else {
			conv.vt.Scale(momentum, conv.vt)
			conv.vt.Add(conv.vt, scaledDW)
		}
		conv.filters.Sub(conv.filters, conv.vt)
	} else {
		conv.filters.Sub(conv.filters, scaledDW)
	}
	return scaledDW
}

func (conv *Conv1D) GetWeights() *mat.Dense {
	return conv.filters
}
