package cellCNN

import (
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/utils"
)

func EncodeLeftForPtMul(L *Matrix, Bcols int, scaling float64, ptL []*ckks.Plaintext, encoder ckks.Encoder, params ckks.Parameters) {

	rows := L.Rows
	cols := L.Cols

	// Diagonalized samples encoding (plaintext)
	var values []complex128
	for i := 0; i < cols>>1; i++ {
		values = make([]complex128, params.Slots())

		m := L.M
		// Each diagonal value
		for j := 0; j < rows; j++ {

			idx0 := j * cols
			idx1 := i*2 + j*cols + j

			cReal := real(m[idx0+idx1%cols])
			cImag := real(m[idx0+(idx1+1)%cols])

			// replicates the value #filters time
			for k := 0; k < Bcols; k++ {
				values[j*Bcols+k] = complex(cReal, -cImag) * complex(scaling, 0)
			}
		}

		encoder.EncodeNTT(ptL[i], values, params.LogSlots())
	}
}

func EncryptLeftForCtMul(L *Matrix, Bcols int, scaling float64, ctL []*ckks.Ciphertext, encoder ckks.Encoder, encryptor ckks.Encryptor, params ckks.Parameters) {

	rows := L.Rows
	cols := L.Cols

	ptL := ckks.NewPlaintext(params, ctL[0].Level(), params.Scale())

	// Diagonalized samples encoding (plaintext)
	var values []complex128
	for i := 0; i < cols>>1; i++ {
		values = make([]complex128, params.Slots())

		m := L.M
		// Each diagonal value
		for j := 0; j < rows; j++ {

			idx0 := j * cols
			idx1 := i*2 + j*cols + j

			cReal := real(m[idx0+idx1%cols])
			cImag := real(m[idx0+(idx1+1)%cols])

			// replicates the value #filters time
			for k := 0; k < Bcols; k++ {
				values[j*Bcols+k] = complex(cReal, -cImag) * complex(scaling, 0)
			}
		}

		encoder.EncodeNTT(ptL, values, params.LogSlots())
		encryptor.Encrypt(ptL, ctL[i])
	}
}

func EncryptRightForPtMul(C *Matrix, nbMatrices, cells int, params ckks.Parameters, level int, encoder ckks.Encoder, encryptor ckks.Encryptor) *ckks.Ciphertext {

	features := C.Rows
	filters := C.Cols

	values := make([]complex128, params.Slots())

	convolutionMatrixSize := ConvolutionMatrixSize(nbMatrices*cells, features, filters)

	// Replicates nbMatrices * (cells * filters) + (features/2)*2*filters elements for the rotations and an additional "#filters" element for the complex trick
	for i := 0; i < convolutionMatrixSize; i++ {
		values[i] = C.M[i%len(C.M)]
	}

	ptC := ckks.NewPlaintext(params, level, params.Scale())
	encoder.EncodeNTT(ptC, values, params.LogSlots())
	ctC := encryptor.EncryptNew(ptC)

	return ctC
}

func MulMatrixLeftPtWithRightCt(A []*ckks.Plaintext, B *ckks.Ciphertext, BRows, BCols int, eval ckks.Evaluator) (AB *ckks.Ciphertext) {

	level := utils.MinInt(A[0].Level(), B.Level())

	for B.Level() > level {
		eval.DropLevel(B, 1)
	}

	scale := B.Scale()

	// Imaginary pre-processing
	tmp := eval.MultByiNew(B)
	eval.Rotate(tmp, BCols, tmp)
	eval.Add(tmp, B, tmp)

	// Pre-rotated the ciphertext
	rotHoisted := []int{}
	for i := 1; i < BRows>>1; i++ {
		rotHoisted = append(rotHoisted, 2*BCols*i)
	}

	BRot := eval.RotateHoisted(tmp, rotHoisted)

	// MultSum
	AB = eval.MulNew(tmp, A[0])
	for i := 1; i < BRows>>1; i++ {
		eval.Mul(BRot[2*BCols*i], A[i], tmp)
		eval.Add(AB, tmp, AB)
	}

	eval.Rescale(AB, scale, AB)

	return AB
}

func MulMatrixLeftCtWithRightCt(A []*ckks.Ciphertext, B *ckks.Ciphertext, BRows, BCols int, eval ckks.Evaluator) (AB *ckks.Ciphertext) {
	level := utils.MinInt(A[0].Level(), B.Level())

	for B.Level() > level {
		eval.DropLevel(B, 1)
	}

	scale := B.Scale()

	// Imaginary pre-processing
	tmp := eval.MultByiNew(B)
	eval.Rotate(tmp, BCols, tmp)
	eval.Add(tmp, B, tmp)

	// Pre-rotated the ciphertext
	rotHoisted := []int{}
	for i := 1; i < BRows>>1; i++ {
		rotHoisted = append(rotHoisted, 2*BCols*i)
	}

	BRot := eval.RotateHoisted(tmp, rotHoisted)

	// MultSum
	AB = eval.MulNew(tmp, A[0])
	for i := 1; i < BRows>>1; i++ {
		eval.Mul(BRot[2*BCols*i], A[i], tmp)
		eval.Add(AB, tmp, AB)
	}

	eval.Rescale(AB, scale, AB)
	eval.Relinearize(AB, AB)

	return AB
}
