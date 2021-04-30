package utils

import "github.com/ldsec/lattigo/v2/ckks"

func Float64ToOneHotEncode(label float64, nfilters int, params *ckks.Parameters, encoder ckks.Encoder) *ckks.Plaintext {
	res := make([]complex128, params.Slots())
	res[int(label)*nfilters] = complex(1, 0)
	encoded := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), res, params.LogSlots())
	return encoded
}

func GenSliceWithOneAt(length int, ind []int) []complex128 {
	res := make([]complex128, length)
	for _, id := range ind {
		res[id] = complex(1, 0)
	}
	return res
}

func GenSliceWithAnyAt(length int, ind []int, any float64) []complex128 {
	res := make([]complex128, length)
	for _, id := range ind {
		res[id] = complex(any, 0)
	}
	return res
}

func GenExtentionInds(from int, to []int) []int {
	res := make([]int, len(to))
	for i, v := range to {
		res[i] = from - v
	}
	return res
}

// default the matrix is row packed into a vector
// recover the matrix in normal format (r*c) then gen the transform matrix filled with (0, 1)
// returns a matrix with Dims: cols * rows
func GenTransposeMatrix(numOfSlots int, r int, c int, inRowPacked bool, ouRowPacked bool) [][]complex128 {
	mt := make([][]complex128, numOfSlots)
	var rid, cid, newi int

	// the i-th element is in (rid, cid)
	for i := 0; i < numOfSlots; i++ {

		if i < r*c {
			// 1. compute the normal ind of the element in matrix
			if inRowPacked {
				rid = i / c
				cid = i % c
			} else {
				rid = i % r
				cid = i / r
			}

			// 2. put this element from (rid, cid) to (cid, rid)
			if ouRowPacked {
				newi = cid*r + rid
			} else {
				newi = rid*c + cid
			}

			// 3. generate rotation vector
			mt[i] = GenSliceWithOneAt(numOfSlots, []int{newi})
		} else {
			mt[i] = GenSliceWithOneAt(numOfSlots, []int{})
		}
	}

	return mt
}

// input colMatrix has Dims: cols, rows
func GenTransposeMap(colMatrix [][]complex128) map[int][]complex128 {
	cols := len(colMatrix)
	rows := len(colMatrix[0])
	diags := make(map[int][]complex128)
	for cid := 0; cid < cols; cid++ {

		oneDiag := make([]complex128, rows)
		remove := true
		for j := 0; j < rows; j++ {
			ptr := (cid + j) % cols
			oneDiag[j] = colMatrix[ptr][j]
			if real(oneDiag[j]) == 1 {
				remove = false
			}
		}
		if !remove {
			diags[cid] = oneDiag
		}
	}

	return diags
}