package cellCNN


import (
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/utils"
	"math"
	"bufio"
	"log"
	"os"
	"strconv"
	"strings"
	"fmt"
)

func WeightsInit(rows, cols, inputs int)(m *ckks.Matrix){
   m = ckks.NewMatrix(rows, cols)
   for i := range m.M {
      m.M[i] = complex(utils.RandFloat64(-1, 1) / math.Sqrt(float64(inputs)), 0)
   }
   return 
}


func LoadTrainDataFrom(path string, samples, cells, features int) (X, Y []*ckks.Matrix) {
	y := String_to_float(Load_file(path+"y_train.txt", samples))
	X = make([]*ckks.Matrix, samples)
	Y = make([]*ckks.Matrix, samples)
	var fname string
	for i := 0; i < samples; i++{
		fname = fmt.Sprintf("X_train/%d.txt", i)
		X[i] = Convert_X_cellCNN(Load_file(path+fname, cells), cells, features)

		Y[i] = ckks.NewMatrix(1, 2)

		if y[i] == 1{
			Y[i].M[0] = 1
		}else{
			Y[i].M[1] = 1
		}
	}
	return X, Y
}

func LoadValidDataFrom(path string, samples, cells, features int) (X, Y []*ckks.Matrix) {
	y := String_to_float(Load_file(path+"y_valid.txt", samples))
	X = make([]*ckks.Matrix, samples)
	Y = make([]*ckks.Matrix, samples)
	var fname string
	for i := 0; i < samples; i++{
		fname = fmt.Sprintf("X_valid/%d.txt", i)
		X[i] = Convert_X_cellCNN(Load_file(path+fname, cells), cells, features)

		Y[i] = ckks.NewMatrix(1, 2)

		if y[i] == 1{
			Y[i].M[0] = 1
		}else{
			Y[i].M[1] = 1
		}
	}
	return X, Y
}


func Convert_X_cellCNN(X []string, cells, features int) (XMat *ckks.Matrix) {

	XMat = ckks.NewMatrix(features, cells)
	for j := 0; j < features; j++ {
		col := strings.Split(X[j], " ")
		copy(XMat.M[j*cells:(j+1)*cells], String_to_float(col))
	}

	XMat = XMat.Transpose()

	return XMat
}


// Load_file load file containing nsamples lines into slice of nsamples strings
func Load_file(fname string, nsamples int) []string {
	output := make([]string, nsamples)
	i := 0
	file, err := os.Open(fname)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		output[i] = scanner.Text()
		i++
		if i >= nsamples {
			break
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	return output
}

// String_to_float convert slice of string to slice of float
func String_to_float(a []string) []complex128 {
	c := make([]complex128, len(a))
	for i := 0; i < len(a); i++{
		tmp, _ := strconv.ParseFloat(a[i], 64)
		c[i] = complex(tmp, 0)
	}
	return c
}

