package leastsquares

import "errors"

// GetCoefficients returns the coefficients for the sigmoid least-squares at degree and interval [-a, a]
func GetCoefficients(degree uint, a float64) ([]float64, error) {
	if a < 0 {
		return nil, errors.New("interval need to be positive")
	}

	ret, ok := leastSquares[leastSquaresKey{degree, uint(a)}]
	if !ok {
		return nil, errors.New("unknown degree and interval combination")
	}

	return ret, nil
}
