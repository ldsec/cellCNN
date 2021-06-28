package utils

import "gonum.org/v1/gonum/mat"

// CompareDenseWeights debug use only, return the mean and mse error
func CompareDenseWeights(ew []complex128, pw *mat.Dense, nfilters, nclasses int) (mean, mse float64) {
	nslots := len(ew)
	meanSum := complex(0, 0)
	mseSum := complex(0, 0)
	for i := 0; i < nclasses; i++ {
		baseline := RetrieveCol(pw, i)
		baseline = append(baseline, make([]complex128, nslots-nfilters)...)
		sub := SliceAdd(ew, baseline, true)
		tmpMean := SliceSum(sub, nfilters)
		tmpMse := SlicePow(SliceCmplxToFloat64(sub), 2)
		meanSum += tmpMean
		mseSum += SliceSum(SliceFloat64ToCmplx(tmpMse), nfilters)
	}
	mean = real(meanSum) / float64(nfilters*nclasses)
	mse = real(mseSum) / float64(nfilters*nclasses)
	return mean, mse
}

// CompareConv1dWeights debug use only, return the mean and mse error
func CompareConv1dWeights(ew [][]complex128, pw *mat.Dense, nmakers, nfilters int) (mean, mse float64) {
	nslots := len(ew[0])
	meanSum := complex(0, 0)
	mseSum := complex(0, 0)
	for i := 0; i < nfilters; i++ {
		baseline := RetrieveCol(pw, i)
		baseline = append(baseline, make([]complex128, nslots-nmakers)...)
		sub := SliceAdd(ew[i], baseline, true)
		tmpMean := SliceSum(sub, nmakers)
		tmpMse := SlicePow(SliceCmplxToFloat64(sub), 2)
		meanSum += tmpMean
		mseSum += SliceSum(SliceFloat64ToCmplx(tmpMse), nmakers)
	}
	mean = real(meanSum) / float64(nfilters*nmakers)
	mse = real(mseSum) / float64(nfilters*nmakers)
	return mean, mse
}
