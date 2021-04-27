package layers

import (
	"github.com/ldsec/cellCNN/semester_project_shufan/claire/utils"
	"gonum.org/v1/gonum/mat"
)

// average pooling layer
type Pool struct {
	ncells   int
	nfilters int
}

// Forward pass of the Pool layer
func (pool *Pool) Forward(input []*mat.Dense) *mat.Dense {

	nsamples := len(input)
	ncells, nfilters := input[0].Dims()
	pool.ncells = ncells
	pool.nfilters = nfilters

	output := mat.NewDense(nsamples, nfilters, nil)
	output_row := make([]float64, pool.nfilters)

	for i := range input {
		for j := 0; j < nfilters; j++ {
			output_row[j] = utils.Mean(mat.Col(nil, j, input[i]))
		}
		output.SetRow(i, output_row)
	}

	return output // nsamples x nfilters
}

// Backward propagation of the Pool layer, given loss/error
func (pool *Pool) Backward(error *mat.Dense) []*mat.Dense {
	// error has dims  nsamples x nfilters
	// next_error has dims nsamples x ncells x nfilters

	nsamples, _ := error.Dims()
	next_error := make([]*mat.Dense, nsamples)

	for i := 0; i < nsamples; i++ {
		//fmt.Println(error)
		err_row := utils.Scale(1/float64(pool.ncells), error.RawRowView(i))

		next_error[i] = mat.NewDense(pool.ncells, pool.nfilters, nil)
		for j := 0; j < pool.ncells; j++ {
			next_error[i].SetRow(j, err_row)
		}
	}
	// no update for pooling layer

	return next_error
}
