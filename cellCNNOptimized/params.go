package cellCNN

import (
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"math"
)


var ThreadsCount = 4

// Number of samples per batch
// MUST BE AN EVEN NUMBER
var BatchSize = 25

// number of samples
// MUST BE AN EVEN NUMBER
var Samples = 1000

// number of cells per sample
// MUST BE AN EVEN NUMBER
var Cells = 200

// number of features
// MUST BE AN EVEN NUMBER
var Features = 16

// number of filters
// MUST BE AN EVEN NUMBER
var Filters = 8

// number of classes
var Classes = 3

// learning rate
var LearningRate = 0.1

// momentum
var Momentum = 0.9

// ring dimension
var LogN = 15

var LogSlots = LogN - 1

var Scale = float64(1 << 52)

// Total number of levels
var Levels = 10

// number of special primes for the key-switching
var NbPi = 2

func ConvolutionMatrixSize(cells, features, filters int) int {
	//     original result   additional rotations padd   complex trick
	return cells*filters + (features/2-1)*2*filters + filters
}

func DenseMatrixSize(filters, classes int) int {
	return filters * classes
}

// GenParams generates CKKS parameters based on the input scale to
// ensure a secure bootstrapping and appropriate moduli
func GenParams() (params ckks.Parameters) {

	var err error

	parametersLiteral := new(ckks.ParametersLiteral)

	log2Scale := math.Log2(Scale)

	bootstrappModuliSize := int(math.Ceil((128.0 + log2Scale) / 3.0))

	if bootstrappModuliSize > 60 {
		panic("scale too high")
	}

	logQi := make([]int, Levels)

	for i := 0; i < 3; i++ {
		logQi[i] = bootstrappModuliSize
	}

	for i := 3; i < Levels; i++ {
		logQi[i] = int(log2Scale)
	}

	logPi := make([]int, NbPi)
	for i := 0; i < NbPi; i++ {
		logPi[i] = bootstrappModuliSize + 1
	}

	parametersLiteral.LogN = LogN
	parametersLiteral.LogQ = logQi
	parametersLiteral.LogP = logPi
	parametersLiteral.LogSlots = LogSlots
	parametersLiteral.Sigma = rlwe.DefaultSigma
	parametersLiteral.Scale = Scale

	if params, err = ckks.NewParametersFromLiteral(*parametersLiteral); err != nil {
		panic(err)
	}

	return

}
