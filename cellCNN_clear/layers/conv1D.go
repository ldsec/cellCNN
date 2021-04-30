package layers

import (
	"github.com/ldsec/cellCNN/cellCNN_clear/utils"
	"gonum.org/v1/gonum/mat"
	"fmt"
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
		//c := utils.WeightsInit(nmarkers*conv.Nfilters, float64(nmarkers))
		c := []float64{-0.157606, -0.042193, 0.090079, -0.140268, 0.127641, -0.163260, 
		0.123652, 0.045518, -0.080589, -0.145210, -0.016246, 0.038132, 
		0.017790, -0.064759, 0.161123, -0.048306, 0.139825, -0.149952, 
		-0.084128, 0.028893, 0.023134, -0.009103, 0.060570, -0.034424, 
		-0.157708, -0.001613, -0.019875, -0.115267, -0.005837, -0.016422, 
		0.009624, -0.005860, -0.163596, 0.054042, -0.109609, 0.154910, 
		-0.105088, -0.113516, -0.001531, -0.013902, -0.155277, -0.026212, 
		0.103340, 0.051511, -0.045778, 0.060698, -0.162107, -0.016309, 
		0.016753, 0.008481, 0.117370, -0.009266, -0.074229, -0.086144, 
		-0.007862, 0.033828, -0.160092, 0.034801, 0.128095, 0.127478, 
		0.012857, 0.086050, -0.101835, -0.153228, 0.034571, -0.082220, 
		-0.010461, 0.132321, 0.092869, 0.019801, 0.132878, -0.053578, 
		-0.099425, 0.003100, 0.123340, 0.015970, 0.067966, 0.086562, 
		0.099241, -0.131986, -0.123341, 0.114430, 0.056390, -0.082455, 
		0.154547, 0.139452, -0.014389, 0.130446, 0.116000, -0.083244, 
		-0.072166, -0.031328, -0.099770, 0.095270, 0.004664, 0.142161, 
		0.088048, 0.050235, -0.004322, 0.049053, 0.023691, 0.077500, 
		0.069644, -0.043004, -0.130544, 0.088716, 0.070841, -0.117993, 
		-0.043149, -0.072074, 0.138816, -0.011218, -0.067313, 0.043315, 
		-0.039192, -0.091592, 0.080425, -0.013404, 0.161842, -0.048344, 
		-0.076622, -0.128067, -0.065961, -0.152914, -0.099298, -0.030963, 
		-0.083517, 0.114592, 0.017944, 0.078460, 0.039129, -0.083513, 
		0.154834, -0.081162, -0.126272, -0.135852, -0.109560, 0.123894, 
		0.035337, 0.056845, 0.005921, 0.141798, -0.149847, 0.161548, 
		0.071480, 0.078476, -0.122914, 0.157709, 0.036181, 0.141595, 
		0.070096, 0.064475, 0.128670, 0.155279, 0.011736, -0.164367, 
		-0.161145, -0.099134, 0.141477, -0.140490, -0.123246, -0.090162, 
		0.053416, -0.046566, 0.014807, -0.153300, -0.142962, 0.135505, 
		0.143485, -0.078104, -0.034584, 0.066045, -0.105522, 0.149281, 
		0.068742, -0.053235, -0.128386, -0.037187, 0.088665, -0.105581, 
		-0.054910, -0.088041, -0.097207, 0.047082, -0.046825, -0.049848, 
		0.085329, 0.013615, -0.091482, 0.056405, -0.139830, 0.049000, 
		0.002704, -0.150111, -0.150939, 0.126627, 0.035670, 0.070295, 
		-0.076432, 0.023660, -0.163961, -0.040700, 0.102170, 0.064880, 
		-0.075007, -0.103649, 0.144220, 0.009432, 0.108479, 0.018681, 
		0.099697, -0.038130, -0.009199, -0.121160, 0.136172, -0.118878, 
		0.045402, 0.114020, 0.059185, 0.041708, 0.046495, 0.117250, }

		conv.filters = mat.NewDense(nmarkers, conv.Nfilters, c)
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

	fmt.Printf(" %v\n", mat.Formatted(conv.filters, mat.Prefix(" "), mat.Excerpt(0)))

	return scaledDW
}

func (conv *Conv1D) GetWeights() *mat.Dense {
	return conv.filters
}
