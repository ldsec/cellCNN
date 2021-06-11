package cellCNN

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"github.com/ldsec/lattigo/v2/utils"
	"math"
)

// Matrix is a struct holding a row flatened complex matrix.
type Matrix struct {
	rows, cols int
	Real       bool
	M          []complex128
}

// NewMatrix creates a new matrix.
func NewMatrix(rows, cols int) (m *Matrix) {
	m = new(Matrix)
	m.M = make([]complex128, rows*cols)
	m.rows = rows
	m.cols = cols
	m.Real = true
	return
}

func (m *Matrix) Copy() (mCopy *Matrix){
	mCopy = new(Matrix)
	mCopy.M = make([]complex128, len(m.M))
	copy(mCopy.M, m.M)
	mCopy.rows = m.rows
	mCopy.cols = m.cols
	mCopy.Real = m.Real
	return
}

func (m *Matrix) Set(rows, cols int, v []complex128){
	m.M = make([]complex128, len(v))
	copy(m.M, v)
	m.rows = rows
	m.cols = cols
}

func (m * Matrix) SetRow(idx int, row []complex128){
	for i := range row{
		m.M[i+idx*m.cols] = row[i]
	}
}

// Rows returns the number of rows of the matrix.
func (m *Matrix) Rows() int {
	return m.rows
}

// Cols returns the number of columns of the matrix.
func (m *Matrix) Cols() int {
	return m.cols
}



// Add adds matrix A and B and stores the result on the target.
func (m *Matrix) Add(A, B *Matrix) {

	if len(A.M) != len(B.M) {
		panic("input matrices are incompatible for addition")
	}

	if m.M == nil {
		m.M = make([]complex128, len(A.M))
	} else if len(m.M) > len(A.M) {
		m.M = m.M[:len(A.M)]
	} else if len(m.M) < len(A.M) {
		m.M = append(m.M, make([]complex128, len(A.M)-len(m.M))...)
	}

	for i := range A.M {
		m.M[i] = A.M[i] + B.M[i]
	}

	if m != A && m != B{
		m.Real = A.Real && B.Real
		m.rows = A.rows
		m.cols = A.cols
	}else if m != B{
		m.rows = B.rows
		m.cols = B.cols
		m.cols = B.cols
	}
}

func (m *Matrix) Abs(){
	for i := range m.M{
		m.M[i] = complex(math.Abs(real(m.M[i])), math.Abs(imag(m.M[i])))
	}
}

func (m *Matrix) Sub(A, B *Matrix) {

	if len(A.M) != len(B.M) {
		panic("input matrices are incompatible for addition")
	}

	if m.M == nil {
		m.M = make([]complex128, len(A.M))
	} else if len(m.M) > len(A.M) {
		m.M = m.M[:len(A.M)]
	} else if len(m.M) < len(A.M) {
		m.M = append(m.M, make([]complex128, len(A.M)-len(m.M))...)
	}

	for i := range A.M {
		m.M[i] = A.M[i] - B.M[i]
	}

	if m != A && m != B{
		m.Real = A.Real && B.Real
		m.rows = A.rows
		m.cols = A.cols
	}else if m != B{
		m.rows = B.rows
		m.cols = B.cols
		m.cols = B.cols
	}
}

func (m *Matrix) SumColumns(A *Matrix){

	rowsA := A.Rows()
	colsA := A.Cols()

	acc := make([]complex128, colsA)

	for i := 0; i < colsA; i++ {
		for j := 0; j < rowsA; j++ {
			acc[i] += A.M[i + j*colsA]
		}
	}

	m.M = acc
	m.rows = 1
	m.cols = colsA
	m.Real = A.Real
}

func (m *Matrix) SumRows(A *Matrix){

	rowsA := A.Rows()
	colsA := A.Cols()

	acc := make([]complex128, rowsA)

	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsA; j++ {
			acc[i] += A.M[i*rowsA + j]
		}
	}

	m.M = acc
	m.rows = rowsA
	m.cols = 1
	m.Real = A.Real
}

func (m *Matrix) Dot(A, B *Matrix){
	if A.Rows() != B.Rows() || A.Cols() != B.Cols() {
		panic("matrices are incompatible for dot product")
	}

	rowsA := A.Rows()
	colsA := A.Cols()

	acc := make([]complex128, rowsA*colsA)

	for i := range A.M{
		acc[i] = A.M[i] * B.M[i]
	}

	m.M = acc
	m.rows = rowsA
	m.cols = colsA
	m.Real = A.Real && B.Real
}

func (m *Matrix) Func(A *Matrix, f func(x complex128)complex128){
	acc := make([]complex128, len(A.M))
	for i := range A.M{
		acc[i] = f(A.M[i])
	}
	m.M = acc
	m.rows = A.Rows()
	m.cols = A.Cols()
	m.Real = A.Real
}

func (m *Matrix) MultConst(A *Matrix,c complex128){
	acc := make([]complex128, len(A.M))
	for i := range A.M{
		acc[i] = c*A.M[i]
	}
	m.M = acc
	m.rows = A.Rows()
	m.cols = A.Cols()
	m.Real = A.Real
}

// MulMat multiplies A with B and returns the result on the target.
func (m *Matrix) MulMat(A, B *Matrix) {

	if A.Cols() != B.Rows() {
		panic("matrices are incompatible for multiplication")
	}

	rowsA := A.Rows()
	colsA := A.Cols()
	colsB := B.Cols()

	acc := make([]complex128, rowsA*colsB)

	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsB; j++ {
			for k := 0; k < colsA; k++ {
				acc[i*colsB+j] += A.M[i*colsA+k] * B.M[j+k*colsB]
			}
		}
	}

	m.M = acc
	m.rows = A.Rows()
	m.cols = B.Cols()

	m.Real = A.Real && B.Real
}

// GenRandomComplexMatrices generates a list of complex matrices.
func GenRandomComplexMatrices(rows, cols, n int) (Matrices []*Matrix) {

	Matrices = make([]*Matrix, n)

	for k := range Matrices {
		m := NewMatrix(rows, cols)
		for i := 0; i < rows*cols; i++ {
			m.M[i] = complex(utils.RandFloat64(-1, 1), utils.RandFloat64(-1, 1))
			m.Real = false
		}
		Matrices[k] = m
	}

	return
}

// GenRandomReaMatrices generates a list of real matrices.
func GenRandomRealMatrices(rows, cols, n int) (Matrices []*Matrix) {

	Matrices = make([]*Matrix, n)

	for k := range Matrices {
		m := NewMatrix(rows, cols)
		for i := 0; i < rows*cols; i++ {
			m.M[i] = complex(utils.RandFloat64(-1, 1), 0)
			m.Real = true
		}
		Matrices[k] = m
	}

	return
}

// GenRandomReaMatrices generates a list of real matrices.
func GenZeroMatrices(rows, cols, n int) (Matrices []*Matrix) {

	Matrices = make([]*Matrix, n)

	for k := range Matrices {
		m := NewMatrix(rows, cols)
		for i := 0; i < rows*cols; i++ {
			m.M[i] = complex(0, 0)
			m.Real = true
		}
		Matrices[k] = m
	}

	return
}

// PermuteRows rotates each row by k where k is the row index.
// Equivalent to Transpoe(PermuteCols(Transpose(M)))
func (m *Matrix) PermuteRows() {
	var index int
	tmp := make([]complex128, m.Cols())
	for i := 0; i < m.Rows(); i++ {
		index = i * m.Cols()
		for j := range tmp {
			tmp[j] = m.M[index+j]
		}

		tmp = append(tmp[i:], tmp[:i]...)

		for j, c := range tmp {
			m.M[index+j] = c
		}
	}
}

// PermuteCols rotates each column by k, where k is the column index.
// Equivalent to Transpoe(PermuteRows(Transpose(M)))
func (m *Matrix) PermuteCols() {
	tmp := make([]complex128, m.Rows())
	for i := 0; i < m.Cols(); i++ {
		for j := range tmp {
			tmp[j] = m.M[i+j*m.Cols()]
		}

		tmp = append(tmp[i:], tmp[:i]...)

		for j, c := range tmp {
			m.M[i+j*m.Cols()] = c
		}
	}
}

// RotateCols rotates each column by k position to the left.
func (m *Matrix) RotateCols(k int) {

	k %= m.Cols()
	var index int
	tmp := make([]complex128, m.Cols())
	for i := 0; i < m.Rows(); i++ {
		index = i * m.Cols()
		for j := range tmp {
			tmp[j] = m.M[index+j]
		}

		tmp = append(tmp[k:], tmp[:k]...)

		for j, c := range tmp {
			m.M[index+j] = c
		}
	}
}

// RotateRows rotates each row by k positions to the left.
func (m *Matrix) RotateRows(k int) {
	k %= m.Rows()
	m.M = append(m.M[k*m.Cols():], m.M[:k*m.Cols()]...)
}

// Transpose transposes the matrix.
func (m *Matrix) Transpose() (mT *Matrix) {
	rows := m.Rows()
	cols := m.Cols()
	mT = NewMatrix(cols, rows)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			mT.M[rows*j+i] = m.M[i*cols+j]
		}
	}
	return
}

// Print prints the target matrix.
func (m *Matrix) Print() {

	if m.Real {
		fmt.Printf("[\n")
		for i := 0; i < m.Rows(); i++ {
			fmt.Printf("[ ")
			for j := 0; j < m.Cols(); j++ {
				fmt.Printf("%11.8f, ", real(m.M[i*m.Cols()+j]))
			}
			fmt.Printf("],\n")
		}
		fmt.Printf("]\n")
	} else {
		fmt.Printf("[")
		for i := 0; i < m.Rows(); i++ {
			fmt.Printf("[ ")
			for j := 0; j < m.Cols(); j++ {
				fmt.Printf("%11.8f, ", m.M[i*m.Cols()+j])
			}
			fmt.Printf("]\n")
		}
		fmt.Printf("]\n")
	}
}

// MarshalBinary serializes a matrix struct to an array of bytes
func MarshalBinary(m *Matrix) ([]byte, error) {
	var b bytes.Buffer
	e := gob.NewEncoder(&b)
	if err := e.Encode(m); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// UnmarshalBinary generates a matrix struct from an array of bytes
func UnmarshalBinary(m *Matrix, bMatrix []byte) error {
	b := bytes.NewBuffer(bMatrix)
	d := gob.NewDecoder(b)
	if err := d.Decode(m); err != nil {
		return err
	}
	return nil
}