package cellCNN

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"github.com/ldsec/lattigo/v2/utils"
	"log"
	"math"
	"math/bits"
	"os"
	"strconv"
	"strings"
)

type PRNGInt struct {
	prng        utils.PRNG
	mask        uint64
	max         uint64
	randomBytes []byte
}

func NewPRNTInt(max int, deterministic bool) (prng *PRNGInt) {
	var err error

	prng = new(PRNGInt)

	if deterministic {
		if prng.prng, err = utils.NewKeyedPRNG(nil); err != nil {
			panic(err)
		}
	} else {
		if prng.prng, err = utils.NewPRNG(); err != nil {
			panic(err)
		}
	}

	prng.mask = uint64(1<<bits.Len64(uint64(max))) - 1
	prng.max = uint64(max)
	prng.randomBytes = make([]byte, 8)
	return prng
}

func (prng *PRNGInt) RandInt() int {

	var c uint64

	mask := prng.mask
	max := prng.max

	prng.prng.Clock(prng.randomBytes)
	c = binary.BigEndian.Uint64(prng.randomBytes) & mask

	for c >= max {
		prng.prng.Clock(prng.randomBytes)
		c = binary.BigEndian.Uint64(prng.randomBytes) & mask
	}

	return int(c)
}

func WeightsInit(rows, cols, inputs int) (m *Matrix) {
	m = NewMatrix(rows, cols)

	for i := range m.M {
		m.M[i] = complex(utils.RandFloat64(-1, 1)/math.Sqrt(float64(inputs)), 0)
	}
	return
}

// GenerateFakeData generate empty/0 X and Y matrices with the correct dimensions
func GenerateFakeData(samples, cells, features int) (X, Y []*Matrix) {
	X = make([]*Matrix, samples)
	Y = make([]*Matrix, samples)

	for i := 0; i < samples; i++ {
		X[i] = NewMatrix(cells, features)
		Y[i] = NewMatrix(1, 2)
	}
	return X, Y
}

func LoadTrainDataFrom(path string, samples, cells, features, labels, typeData int) (X, Y []*Matrix) {
	y := String_to_float(Load_file(path+"y_train.txt", samples))
	X = make([]*Matrix, samples)
	Y = make([]*Matrix, samples)
	var fname string
	if typeData != 1 {
		for i := 0; i < samples; i++ {
			fname = fmt.Sprintf("X_train/%d.txt", i)
			X[i] = Convert_X_cellCNN(Load_file(path+fname, cells), cells, features)

			Y[i] = NewMatrix(1, labels)

			Y[i].M[int(real(y[i]))] = 1
		}
	}

	if typeData == 1 {
		fname = fmt.Sprintf("X_train/all.txt")
		Xtemp := Load_file(path+fname, cells)
		ind := 0
		Xtemp2 := Convert_X_cellCNN(Xtemp, cells*samples, features)

		for i := 0; i < samples; i++ {
			row := Xtemp2.M[ind:(ind + (cells * features))]
			//fmt.Println(row)
			X[i] = NewMatrix(cells, features)

			X[i].M = row

			Y[i] = NewMatrix(1, labels)

			Y[i].M[int(real(y[i]))] = 1
			ind = ind + (cells * features)
		}
	}
	return X, Y
}

func LoadValidDataFrom(path string, samples, cells, features, labels int) (X, Y []*Matrix) {
	y := String_to_float(Load_file(path+"y_valid.txt", samples))
	X = make([]*Matrix, samples)
	Y = make([]*Matrix, samples)
	var fname string
	for i := 0; i < samples; i++ {
		fname = fmt.Sprintf("X_valid/%d.txt", i)
		X[i] = Convert_X_cellCNN(Load_file(path+fname, cells), cells, features)

		Y[i] = NewMatrix(1, labels)

		Y[i].M[int(real(y[i]))] = 1
	}
	return X, Y
}

func Convert_X_cellCNN(X []string, cells, features int) (XMat *Matrix) {

	XMat = NewMatrix(features, cells)
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
	for i := 0; i < len(a); i++ {
		tmp, _ := strconv.ParseFloat(a[i], 64)
		c[i] = complex(tmp, 0)
	}
	return c
}
