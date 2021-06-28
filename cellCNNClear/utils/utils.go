package utils

import (
	"go.dedis.ch/onet/v3/log"
	"math"
	"math/rand"
)

// ********************************** SLICE MANIPULATION **********************************
// Random generate a random floating point number between a and b
func Random(a, b float64) float64 {
	return (b-a)*rand.Float64() + a
}

//Add element-wise addition
func Add(a []float64, b []float64) []float64 {
	l := int(math.Min(float64(len(a)), float64(len(b))))
	c := make([]float64, l)
	for i := range c {
		c[i] = a[i] + b[i]
	}
	return c
}

// Relu is the original (non-approximated) relu (rectifier) function: max(0,x)
func Relu(x float64) float64 {
	return math.Max(0, x)
}

// ReluD is the derivative of the Relu function
// {0: if x < 0, 1: if x > 0
func ReluD(x float64) float64 {
	if x > 0 {
		return 1.0
	} else if x < 0 {
		return 0.0
	} else {
		log.Warn("ReluD is undefined for x==0")
		return 0.0
	}
}

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
func SigmoidDApproxClear(z float64, a float64, coeffs []float64) (float64, error) {

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

// Sub element-wise subtraction
func Sub(a []float64, b []float64) []float64 {
	l := int(math.Min(float64(len(a)), float64(len(b))))
	c := make([]float64, l)
	for i := 0; i < l; i++ {
		c[i] = a[i] - b[i]
	}
	return c
}

func Max(a []float64) float64 {
	max := a[0]
	for i := range a {
		if a[i] > max {
			max = a[i]
		}
	}
	return max
}

func Min(a []float64) float64 {
	max := a[0]
	for i := range a {
		if a[i] < max {
			max = a[i]
		}
	}
	return max
}

// Scale scale every element of a by k
func Scale(k float64, a []float64) []float64 {
	c := make([]float64, len(a))
	for i := range c {
		c[i] = k * a[i]
	}
	return c
}

// Mul element-wise multiplication of a and b
func Mul(a []float64, b []float64) []float64 {
	l := int(math.Min(float64(len(a)), float64(len(b))))
	c := make([]float64, l)
	for i := range c {
		c[i] = a[i] * b[i]
	}
	return c
}

// Apply apply f to each value of slice a
func Apply(f func(float64) float64, a []float64) []float64 {
	c := make([]float64, len(a))
	for i := range c {
		c[i] = f(a[i])
	}
	return c
}

// Fill returns slice of given length filled with value
func Fill(length int, value float64) []float64 {
	c := make([]float64, length)
	for i := range c {
		c[i] = value
	}
	return c
}

// FillNorm returns slice of given length filled with values taken from the normal distrib
func FillNorm(length int, mean float64, std float64) []float64 {
	a := make([]float64, length)
	for i := range a {
		a[i] = rand.NormFloat64()*std + mean
	}
	return a
}

func WeightsInit(length int, inputs float64) []float64 {
	a := make([]float64, length)
	for i := range a {
		a[i] = Random(-1, 1) / math.Sqrt(inputs)
		//a[i] = Random(-0.05, 0.05)
	}
	return a
}

//InnerSum compute inner sum
func InnerSum(a []float64, b []float64) float64 {
	l := int(math.Min(float64(len(a)), float64(len(b))))
	c := 0.
	for i := 0; i < l; i++ {
		c = c + a[i]*b[i]
	}
	return c
}

func Mean(a []float64) float64 {
	c := 0.
	for i := range a {
		c = c + a[i]
	}
	return c / float64(len(a))
}

// SoftMax activation function (non-approximated)
func Softmax(x []float64) []float64 {
	out := make([]float64, len(x))
	var sum float64
	max := Max(x)
	for i, y := range x {
		out[i] = math.Exp(y - max)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// ********************************** ACTIVATION FUNCTIONS **********************************

// indicator: returns 1{x > 0}
func Indic(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// identity
func Id(x float64) float64 {
	return x
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Sigmoid function shifted so that f(x) is in [-1; 1]
func Sigmoid2(x float64) float64 {
	return 2*Sigmoid(x) - 1
}

// make function float -> float into function to apply to matrix (int, int, float -> float)
func ToApply(f func(float64) float64) func(i, j int, v float64) float64 {
	f_prime := func(i, j int, v float64) float64 {
		return f(v)
	}
	return f_prime
}
