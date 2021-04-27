package layers

// import (
// 	"fmt"
// 	"time"

// 	"github.com/ldsec/lattigo/v2/ckks"
// )

// // requires rotation keys for:
// // 	[1, ncells-1] * nmakers
// // 	slots - [1, nfilters-1]

// // Pool average pooling layer
// type Pool struct {
// 	ncells   int
// 	nmakers  int
// 	nfilters int
// 	slots    uint64 // num of slots in params
// }

// // NewPool constructor
// func NewPool(ncells int, nmakers int, nfilters int) *Pool {
// 	return &Pool{
// 		ncells:   ncells,
// 		nmakers:  nmakers,
// 		nfilters: nfilters,
// 	}
// }

// // Forward pass of the Pool layer
// func (pool *Pool) Forward(
// 	input []*ckks.Ciphertext,
// 	evaluator ckks.Evaluator,
// 	rotKeys *ckks.RotationKeys,
// 	mask *ckks.Plaintext,
// 	slots uint64,
// ) *ckks.Ciphertext {

// 	var sums *ckks.Ciphertext
// 	var output *ckks.Ciphertext

// 	// tmpRecords := make([]*ckks.Ciphertext, len(input))

// 	// wg := sync.WaitGroup{}

// 	// for i := 0; i < len(input); i++ {
// 	// 	wg.Add(1)
// 	// 	go func(i int, evaluator ckks.Evaluator, rotKeys *ckks.RotationKeys, cp *ckks.Ciphertext) {
// 	// 		defer wg.Done()
// 	// 		sums := cp

// 	// 		step := uint64(pool.nmakers)
// 	// 		rotSlice := NewSlice(uint64(pool.nmakers), uint64(pool.nmakers*(pool.ncells-1)), step)
// 	// 		fmt.Println("rot slice: ", rotSlice)
// 	// 		shiftMap := evaluator.RotateHoisted(cp, rotSlice, rotKeys)

// 	// 		for _, v := range shiftMap {
// 	// 			sums = evaluator.AddNew(sums, v)
// 	// 		}

// 	// 		// for j := 1; j < pool.ncells; j++ {
// 	// 		// 	rotVal = evaluator.RotateNew(input[i], uint64(j*pool.nmakers), rotKeys)
// 	// 		// 	sums = evaluator.AddNew(sums, rotVal)
// 	// 		// }
// 	// 		// 2. divide the result by ncells.
// 	// 		sums.MulScale(float64(pool.ncells))
// 	// 		// 3. mask the ouput to keep only the left most element
// 	// 		sums = evaluator.MulRelinNew(sums, mask, nil)
// 	// 		// output = append(output, sums)
// 	// 		// 4.collect all the ciphertext into one ciphertext by rotation

// 	// 		if i != 0 {
// 	// 			sums = evaluator.RotateNew(sums, slots-uint64(i), rotKeys) // requires rotKeys from (slot-1) to (slots - nfilters)
// 	// 		}
// 	// 		tmpRecords[i] = sums
// 	// 	}(i, evaluator, rotKeys, input[i])
// 	// }

// 	// wg.Wait()

// 	// output = tmpRecords[0]
// 	// for i := 1; i < len(tmpRecords); i++ {
// 	// 	output = evaluator.AddNew(output, tmpRecords[i])
// 	// }

// 	// return output

// 	// 1. rotate for j*nmakers and add
// 	for i := 0; i < len(input); i++ {
// 		t1 := time.Now()
// 		sums = input[i]
// 		for j := 1; j < pool.ncells; j++ {
// 			rotVal := evaluator.RotateNew(input[i], uint64(j*pool.nmakers), rotKeys)
// 			sums = evaluator.AddNew(sums, rotVal)
// 		}

// 		// 2. divide the result by ncells.
// 		sums.MulScale(float64(pool.ncells))
// 		// 3. mask the ouput to keep only the left most element
// 		sums = evaluator.MulRelinNew(sums, mask, nil)
// 		// output = append(output, sums)
// 		// 4.collect all the ciphertext into one ciphertext by rotation
// 		if i == 0 {
// 			output = sums
// 		} else {
// 			sums = evaluator.RotateNew(sums, slots-uint64(i), rotKeys) // requires rotKeys from (slot-1) to (slots - nfilters)
// 			output = evaluator.AddNew(output, sums)
// 		}
// 		fmt.Printf("Round %v pooling with time: %v\n", i, time.Since(t1).Seconds())
// 	}
// 	return output // 1 x nfilters
// }
