package utils

import (
	"fmt"
	"testing"
)

func TestPlainRotation(t *testing.T) {
	arr := make([]complex128, 10)
	for i := 0; i < 10; i++ {
		arr[i] = complex(float64(i), 0)
	}
	k1 := 3
	k2 := -3
	fmt.Println(k2 % 2)
	fmt.Printf("\n##### Slice Rotation: #####\n")
	fmt.Printf("Original: %v\nAfter Rotate %d Get: %v\n", arr, k1, SliceRotation(arr, k1))
	fmt.Printf("Original: %v\nAfter Rotate %d Get: %v\n", arr, k2, SliceRotation(arr, k2))
}

func TestPlainMult(t *testing.T) {
	arr1 := make([]complex128, 10)
	for i := 0; i < 10; i++ {
		arr1[i] = complex(float64(i), 0)
	}

	arr2 := make([]complex128, 10)
	for i := 0; i < 10; i++ {
		arr2[i] = complex(float64(i)+1, 0)
	}
	fmt.Printf("\n##### Slice mult: #####\n left: %v\n right: %v\n res: %v\n", arr1, arr2, SliceMult(arr1, arr2))
}

func TestPlainCollect(t *testing.T) {
	arr := make([]complex128, 10)
	for i := 0; i < 10; i++ {
		arr[i] = complex(float64(i), 0)
	}
	inds := []int{1, 3, 6, 9}
	fmt.Printf("\n##### Slice collect: #####\n arr: %v\n inds: %v\n res: %v\n", arr, inds, SliceCollect(arr, inds))

}

func TestPlainTranspose(t *testing.T) {
	arr := make([]complex128, 6)
	for i := 0; i < len(arr); i++ {
		arr[i] = complex(float64(i), 0)
	}
	col := 2
	row := 3
	rowPacked := false
	fmt.Printf(
		"\n##### Slice transpose: #####\n arr: %v, rowPakced: %v, row: %d, col: %d, \n res: %v\n",
		arr, rowPacked, row, col, SliceTranspose(arr, col, row, rowPacked),
	)

}

func TestPlainReplicate(t *testing.T) {
	arr := make([]complex128, 20)
	n := 5
	rep := 3
	for i := 0; i < n; i++ {
		arr[i] = complex(float64(i), 0)
	}

	fmt.Printf(
		"\n##### Slice replicate: #####\n arr: %v, rep: %d, \n res: %v\n",
		arr, rep, SliceReplicate(arr, n, rep),
	)
}

func TestPlainInnerSum(t *testing.T) {
	arr := make([]complex128, 20)
	n := 4
	batch := 5
	for i := 0; i < len(arr); i++ {
		arr[i] = complex(float64(i), 0)
	}

	fmt.Printf(
		"\n##### Slice innerSum: #####\n arr: %v, batch: %d, n: %d \n res: %v\n",
		arr, batch, n, SliceInnerSum(arr, batch, n),
	)
}

func TestPlainEvaluatePoly(t *testing.T) {
	n := 5
	arr := make([]complex128, 10)
	for i := 0; i < n; i++ {
		arr[i] = complex(float64(i%3), 0)
	}
	coeffs := []complex128{3, 2, 1}

	fmt.Printf(
		"\n##### Slice evalute poly: #####\n arr: %v, coeffs: %v, n: %d \n res: %v\n",
		arr, coeffs, n, SliceEvaluatePoly(arr, coeffs, n),
	)
}

func TestPlainSliceExtend(t *testing.T) {
	n := []int{1, 5, 15}
	k := 3
	arr := make([]complex128, 20)
	for i := 0; i < 20; i++ {
		arr[i] = complex(float64(i), 0)
	}

	fmt.Printf(
		"\n##### Slice extend: #####\n arr: %v, n: %d, k: %d \n res: %v\n",
		arr, n, k, SliceExtend(arr, n, k),
	)
}

func TestPlainSliceAdd(t *testing.T) {
	n := 5
	isNegative := false

	arr1 := make([]complex128, n)
	for i := 0; i < n; i++ {
		arr1[i] = complex(float64(i), 0)
	}

	arr2 := make([]complex128, n)
	for i := 0; i < n; i++ {
		arr2[i] = complex(float64(i%2), 0)
	}

	fmt.Printf(
		"\n##### Slice Add: #####\n arr1: %v\n arr2: %v\n res: %v\n",
		arr1, arr2, SliceAdd(arr1, arr2, isNegative),
	)
}

func TestPlainSliceSub(t *testing.T) {
	n := 5
	isNegative := true

	arr1 := make([]complex128, n)
	for i := 0; i < n; i++ {
		arr1[i] = complex(float64(i), 0)
	}

	arr2 := make([]complex128, n)
	for i := 0; i < n; i++ {
		arr2[i] = complex(float64(i%2), 0)
	}

	fmt.Printf(
		"\n##### Slice Add: #####\n arr1: %v\n arr2: %v\n res: %v\n",
		arr1, arr2, SliceAdd(arr1, arr2, isNegative),
	)
}
