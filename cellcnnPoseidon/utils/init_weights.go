package utils

import (
	"math"
	"math/rand"
)

// Random generate a random floating point number between a and b
func Random(a, b float64) float64 {
	return (b-a)*rand.Float64() + a
}

// WeightsInit init a complect slice in (-1,1) with replication option
func WeightsInit(slots int, length int, nmakers float64, rep int) ([]complex128, []complex128) {
	a := make([]complex128, slots)
	if rep == 1 {
		for i := 0; i < length; i++ {
			real := Random(-1, 1) / math.Sqrt(nmakers)
			a[i] = complex(real, 0)
		}
	} else {
		for i := 0; i < length; i++ {
			real := Random(-1, 1) / math.Sqrt(nmakers)
			for j := 0; j < rep; j++ {
				a[i+j*length] = complex(real, 0)
			}
		}
	}
	return a, a[:length]
}
