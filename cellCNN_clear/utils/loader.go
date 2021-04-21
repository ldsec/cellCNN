package utils

import (
	"bufio"
	"gonum.org/v1/gonum/mat"
	"log"
	"os"
	"strconv"
	"strings"
)

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
func String_to_float(a []string) []float64 {
	c := make([]float64, len(a))
	for i := 0; i < len(a); i++ {
		c[i], _ = strconv.ParseFloat(a[i], 64)
	}
	return c
}

func Convert_X_cellCNN(X []string, ncells int, nfeatures int) *mat.Dense {
	r := ncells
	c := nfeatures

	X_mat := mat.NewDense(r, c, nil)
	col := make([]string, r)
	for j := 0; j < c; j++ {
		col = strings.Split(X[j], " ")
		X_mat.SetCol(j, String_to_float(col))
	}

	return X_mat
}
