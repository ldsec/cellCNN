package approx

import (
	"fmt"
	"math"

	"github.com/ldsec/cellCNN/cellcnnPoseidon/approx/leastsquares"
)

//SigmoidApproxClear approximates the sigmoid function within interval [-a,a] with a polynomial degree degree
func SigmoidApproxClear(z float64, a float64, coeffs []float64) float64 {
	sum := coeffs[0]
	pow := 1
	for i := 1; i < len(coeffs); i++ {
		sum += coeffs[i] * math.Pow(z/a, float64(pow))
		pow++
	}
	return sum
}

// SigmoidDApproxClear approximates the sigmoid function within interval [-a,a] with a polynomial degree degree
func SigmoidDApproxClear(z float64, degree uint, a float64) (float64, error) {
	coeffs, err := leastsquares.GetCoefficients(degree, a)
	if err != nil {
		return 0, fmt.Errorf("get coefficients: %v", err)
	}

	//	fmt.Print(coeffs[1]* math.Pow(1/a, float64(1)))
	sum := coeffs[1] * math.Pow(1/a, float64(1))
	pow := 2
	for i := 2; i < len(coeffs); i++ {
		//	fmt.Print(coeffs[i] * math.Pow(1/a, float64(pow))*float64(pow))
		sum += coeffs[i] * math.Pow(1/a, float64(pow)) * math.Pow(z, float64(pow-1)) * float64(pow)
		pow++
	}

	return sum, nil
}
