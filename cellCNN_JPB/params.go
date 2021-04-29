package cellCNN

import (
	"github.com/ldsec/lattigo/v2/ckks"
	"math"
	"fmt"
)

// number of samples
// MUST BE AN EVEN NUMBER
var Samples = 82

// Number of samples per batch
// MUST BE AN EVEN NUMBER
var BatcheSize = 82

// number of cells in each batch
// MUST BE AN EVEN NUMBER
var Cells = 200

// number of features 
// MUST BE AN EVEN NUMBER
var Features = 38

// number of filters
// MUST BE AN EVEN NUMBER
var Filters = 8

// number of classes
var Classes = 2

// learning rate
var LearningRate = 1.0

// momentum
var Momentum = 0.0

// ring dimension
var LogN = 15

var Scale = float64(1 << 52)

// Total number of levels
var Levels = 10

// number of special primes for the key-switching
var NbPi = 2

func ConvolutionMatrixSize(cells, filters, features int) int {
	//     original result   additional rotations padd   complex trick
	return cells * filters + (features/2 -1)*2*filters + filters
}

func DenseMatrixSize(filters, classes int) int{
	return filters * classes
}

// GenParams generates CKKS parameters based on the input scale to
// ensure a secure bootstrapping and appropriate moduli
func GenParams() (params *ckks.Parameters){

	var err error

	log2Scale := math.Log2(Scale)


	bootstrappModuliSize := int(math.Ceil((128.0 + log2Scale)/3.0))

	if bootstrappModuliSize > 60{
		panic("scale too high")
	}

	fmt.Println(bootstrappModuliSize)

	logQi := make([]int, Levels)

	for i := 0; i < 3; i++{
		logQi[i] = bootstrappModuliSize
	}

	for i := 3; i < Levels; i++{
		logQi[i] = int(log2Scale)
	}

	logPi := make([]int, NbPi)
	for i := 0; i < NbPi; i++{
		logPi[i] = bootstrappModuliSize+1
	}

	logModuli := ckks.LogModuli{
		LogQi: logQi,
		LogPi: logPi,
	}

	if params, err = ckks.NewParametersFromLogModuli(LogN, &logModuli); err != nil{
		panic(err)
	}

	params.SetScale(Scale)
	params.SetLogSlots(LogN-1)

	return 

}