package cellCNN

import (
	"github.com/ldsec/lattigo/v2/ckks"
	"math"
	"fmt"
)


// number of cells in each batch
var Cells = 200

// number of features
var Features = 40

// number of filters
var Filters = 8

// number of classes
var Classes = 2

// learning rate
var LearningRate = 1.0

// momentum
var Momentum = 0.9

// ring dimension
var LogN = 15

// number of special primes for the key-switching
var NbPi = 2


// GenParams generates CKKS parameters based on the input scale to
// ensure a secure bootstrapping and appropriate moduli
func GenParams(scale float64) (params *ckks.Parameters){

	var err error

	log2Scale := math.Log2(scale)


	bootstrappModuliSize := int(1+(128.0 + log2Scale)/3.0)

	if bootstrappModuliSize > 60{
		panic("scale too high")
	}

	fmt.Println(bootstrappModuliSize)

	logQi := make([]int, 11)

	for i := 0; i < 3; i++{
		logQi[i] = bootstrappModuliSize
	}

	for i := 3; i < 11; i++{
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

	params.SetScale(scale)
	params.SetLogSlots(LogN-1)

	return 

}