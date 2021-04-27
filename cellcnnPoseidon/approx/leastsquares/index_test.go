package leastsquares_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/ldsec/cellCNN/internal/leastsquares"
)

func TestGetCoeffs(t *testing.T) {
	x, err := leastsquares.GetCoefficients(7, 9)
	require.NoError(t, err)
	require.Equal(t, []float64{0.5, 1.8619, 0, -4.8585, 0, 6.5216, 0, -3.0665}, x)
}

func TestGetCoeffsOutOfRangeDegree(t *testing.T) {
	_, err := leastsquares.GetCoefficients(999, 9)
	require.Error(t, err)
}

func TestGetCoeffsNegativeInterval(t *testing.T) {
	_, err := leastsquares.GetCoefficients(7, -1)
	require.Error(t, err)
}
