package layers

import (
	"fmt"
	"time"

	"github.com/ldsec/cellCNN/cellcnnPoseidon/utils"
	"github.com/ldsec/lattigo/v2/ckks"
)

// Conv1D conduct 1D convolution and pooling together
type Conv1D struct {
	filters   []*ckks.Ciphertext // len(filters) = Nfilters
	lastInput *ckks.Plaintext    // packed ncells after encoding
	// Activation  func(float64) float64
	// dActivation func(float64) float64
	// u           []*mat.Dense
	encoder    ckks.Encoder
	gradient   []*ckks.Ciphertext
	isMomentum bool
	vt         []*ckks.Ciphertext // momentum
}

// NewConv1D constructor
func NewConv1D(filters []*ckks.Ciphertext) *Conv1D {
	return &Conv1D{
		filters:    filters,
		isMomentum: false,
	}
}

func (conv *Conv1D) WithWeights(filters []*ckks.Ciphertext) {
	conv.filters = filters
}

func (conv *Conv1D) WithLastInput(input *ckks.Plaintext) {
	conv.lastInput = input
}

func (conv *Conv1D) WithEncoder(encoder ckks.Encoder) {
	conv.encoder = encoder
}

func (conv *Conv1D) GetGradient() []*ckks.Ciphertext {
	return conv.gradient
}

func (conv *Conv1D) GetWeights() []*ckks.Ciphertext {
	return conv.filters
}

func (conv *Conv1D) SetMomentum() {
	conv.isMomentum = true
}

func (conv *Conv1D) InitRotationInds(sts *CellCnnSettings, kgen ckks.KeyGenerator) []int {
	nmakers := sts.Nmakers
	nfilters := sts.Nfilters
	ncells := sts.Ncells
	// nclasses := sts.Nclasses

	// Conv1D Forward
	// 1. for input weights matrix mult
	Fmult := kgen.GenRotationIndexesForInnerSum(1, nmakers*ncells)
	// 2. for rotation the result to left most slots
	Fshift := make([]int, 0)
	for i := 1; i < nfilters; i++ {
		Fshift = append(Fshift, -i)
	}
	Finds := append(Fmult, Fshift...)

	//Conv1D Backward
	// 3. for replicate the err for each filter
	// ---------------------------------------
	// Brep := make([]int, 0)
	// for i := 0; i < sts.Nfilters; i++ {
	// 	extInds := utils.GenExtentionInds(i*sts.Nclasses, utils.NewSlice(0, sts.Ncells*sts.Nmakers-1, 1))
	// 	Brep = append(Brep, extInds...)
	// }
	// ---------------------------------------
	// +++++++++++++++++++++++++++++++++++++++
	Brep1 := utils.NewSlice(0, (sts.Nfilters-1)*sts.Nclasses, sts.Nclasses)
	Brep2 := kgen.GenRotationIndexesForInnerSum(-1, sts.Ncells*sts.Nmakers)
	Brep := append(Brep1, Brep2...)
	// +++++++++++++++++++++++++++++++++++++++

	// 4. left collect gradient
	Bcol := make([]int, sts.Nmakers)
	for i := 0; i < sts.Nmakers; i++ {
		Bcol[i] = sts.Ncells*i - i
	}
	// 5. replicate gradient
	// ++++++++++++++++++++++++++++
	Brepg := kgen.GenRotationIndexesForInnerSum(-sts.Nmakers, sts.Ncells)
	B0 := kgen.GenRotationIndexesForInnerSum(1, sts.Ncells)
	// ----------------------------
	// Brepg := utils.NegativeSlice(utils.NewSlice(0, (sts.Ncells-1)*sts.Nmakers, sts.Nmakers))
	// ----------------------------
	Binds := append(Brep, Bcol...)
	Binds = append(Binds, Brepg...)
	Binds = append(Binds, B0...)

	Inds := append(Finds, Binds...)

	return Inds
}

func (conv *Conv1D) TransposeInput(sts *CellCnnSettings, input *ckks.Plaintext, params *ckks.Parameters) *ckks.Plaintext {
	slice := conv.encoder.Decode(input, params.LogSlots())
	rowPacked := true
	sliceT := utils.SliceTranspose(slice, sts.Nmakers, sts.Ncells, rowPacked)
	plaintextT := conv.encoder.EncodeNTTAtLvlNew(params.MaxLevel(), sliceT, params.LogSlots())
	return plaintextT
}

// Forward computes a forward pass of the Conv1D layer, using newFilters if not nil
// activation function is identity
func (conv *Conv1D) Forward(
	input *ckks.Plaintext,
	newFilters []*ckks.Ciphertext,
	sts *CellCnnSettings,
	evaluator ckks.Evaluator,
	params *ckks.Parameters,
	mask *ckks.Plaintext,
) *ckks.Ciphertext {

	if newFilters != nil {
		conv.filters = newFilters
	}

	conv.lastInput = input

	var output *ckks.Ciphertext
	batch := 1                    // only the left most
	n := sts.Nmakers * sts.Ncells // num of slots to add together

	actvs := make([]*ckks.Ciphertext, len(conv.filters)) // activations for each filter

	// wg := sync.WaitGroup{}
	// loop over all filters
	for i, filter := range conv.filters {
		eval := evaluator.ShallowCopy()
		// wg.Add(1)
		// go func(i int, filter *ckks.Ciphertext, eval ckks.Evaluator) {
		// defer wg.Done()

		// 1. multiply the filters and the input
		actvs[i] = eval.MulRelinNew(filter, input)

		eval.Rescale(actvs[i], params.Scale(), actvs[i])

		// 2. innerSum to get the pooling result (before avg) in the left most slot
		eval.InnerSum(actvs[i], batch, n, actvs[i])

		// 3. mask the ouput to keep only the left most element
		eval.MulRelin(actvs[i], mask, actvs[i])

		eval.Rescale(actvs[i], params.Scale(), actvs[i])

		// 	}(id, ft, evaluator.ShallowCopy())
	}

	// wg.Wait()

	// output = actvs[0]

	for i := 0; i < len(actvs); i++ {
		// 5. rotate the result to i-th place and add together
		if i == 0 {
			output = actvs[i]
		} else {
			evaluator.Rotate(actvs[i], -i, actvs[i])
			evaluator.Add(output, actvs[i], output)
		}
	}

	// // fmt.Printf("--conv 6.collect to one ciphertext <level : %v, scale: %v>\n", output.Level(), math.Log2(output.Scale()))

	return output
}

func (conv *Conv1D) Backward(
	inErr *ckks.Ciphertext, sts *CellCnnSettings, params *ckks.Parameters,
	evaluator ckks.Evaluator, encoder ckks.Encoder,
) []*ckks.Ciphertext {
	// return dw for each filter, for debug

	// 1. mult the dActv with the income err
	// currently dActv = 1, skip this part

	tx0 := time.Now()

	// 2. extend the input err, pack each slot of err to length ncells*nmakers, with a factor of 1/n
	maskedErrSlice := make([]*ckks.Ciphertext, sts.Nfilters)

	// pooling factor, add to mask
	pf := 1.0 / float64(sts.Ncells)

	//  2.1 mask each valid slot
	for i := 0; i < sts.Nfilters; i++ {
		mask := utils.GenSliceWithAnyAt(params.Slots(), []int{i * sts.Nclasses}, pf)
		maskPlain := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), mask, params.LogSlots())
		maskedErrSlice[i] = evaluator.MulRelinNew(inErr, maskPlain)
		if err := evaluator.Rescale(maskedErrSlice[i], params.Scale(), maskedErrSlice[i]); err != nil {
			panic("fail to rescale, conv backward, err extention")
		}
		// if i == 0 {
		// 	// inErr - 1
		// 	fmt.Printf("inErr - 1 " + utils.PrintCipherLevel(maskedErrSlice[i], params))
		// }
	}

	ty0 := time.Since(tx0)

	fmt.Printf("> Time comsumed in point0 is: %v\n", ty0.Seconds())

	tx1 := time.Now()

	//  2.2 rotate to extend one slots to ncells*nmakers, need to generate new rotation keys
	// e.g.
	// from:
	//     [a, 0, 0, 0, 0,...]
	// to:
	//     [a, a, a, a, 0,...]
	extErrSlice := make([]*ckks.Ciphertext, sts.Nfilters)
	for i, mErr := range maskedErrSlice {
		// left rotate once and right replicate
		leftMostTmp := evaluator.RotateNew(mErr, i*sts.Nclasses)
		evaluator.InnerSum(leftMostTmp, -1, sts.Ncells*sts.Nmakers, leftMostTmp)
		extErrSlice[i] = leftMostTmp
	}

	ty1 := time.Since(tx1) // 18 seconds

	fmt.Printf("> Time comsumed in point1 is: %v\n", ty1.Seconds())

	// 3. mult the extended err with the transposed last input
	dwSlice := make([]*ckks.Ciphertext, sts.Nfilters)

	tx2 := time.Now()

	inputT := conv.TransposeInput(sts, conv.lastInput, params)

	ty2 := time.Since(tx2)

	fmt.Printf("> Time comsumed in point2 is: %v\n", ty2.Seconds())

	batch := 1
	n := sts.Ncells

	tx := time.Now()
	for i, extErr := range extErrSlice {
		tpart1 := time.Now()
		dwSlice[i] = evaluator.MulRelinNew(inputT, extErr)
		if err := evaluator.Rescale(dwSlice[i], params.Scale(), dwSlice[i]); err != nil {
			panic("fail to rescale, conv backward, inputT mult extErr")
		}
		// if i == 0 {
		// 	fmt.Printf("inErr - 2 " + utils.PrintCipherLevel(dwSlice[i], params))
		// }

		// 4. innerSum to get the result in the correct place, valid at: (0~m-1)*n
		evaluator.InnerSum(dwSlice[i], batch, n, dwSlice[i])
		tpart2 := time.Now()
		ts1 := time.Since(tpart1).Seconds()

		// 5. mask & rotate to sum the result in the left most slots
		// require new rotation ids
		dwSlice[i] = utils.MaskAndCollectToLeft(dwSlice[i], params, encoder, evaluator, 0, sts.Ncells, sts.Nmakers)

		tpart3 := time.Now()
		ts2 := time.Since(tpart2).Seconds()
		// 6. replicate ncells times
		evaluator.InnerSum(dwSlice[i], -sts.Nmakers, sts.Ncells, dwSlice[i])
		ts3 := time.Since(tpart3).Seconds()
		tsum := time.Since(tpart1).Seconds()
		if i == 0 {
			fmt.Printf("Sum: %v, part1: %v(%v), part2: %v(%v), part3: %v(%v)\n",
				tsum, ts1, ts1/tsum, ts2, ts2/tsum, ts3, ts3/tsum,
			)
		}
	}
	ty := time.Since(tx)

	fmt.Printf("> Time comsumed in loop is: %v\n", ty.Seconds())

	conv.gradient = dwSlice

	return dwSlice
}

func (conv *Conv1D) Step(lr float64, momentum float64, eval ckks.Evaluator) bool {
	if conv.gradient != nil {
		for i := range conv.filters {
			update := eval.MultByConstNew(conv.gradient[i], lr)
			// if use momentum
			if conv.isMomentum {
				if conv.vt == nil {
					conv.vt[i] = update.CopyNew().Ciphertext()
				} else {
					conv.vt[i] = eval.MultByConstNew(conv.vt[i], momentum)
					conv.vt[i] = eval.AddNew(conv.vt[i], update)
				}
				conv.filters[i] = eval.SubNew(conv.filters[i], conv.vt[i])
			} else {
				// else use only gradient
				conv.filters[i] = eval.SubNew(conv.filters[i], update)
			}
		}
		return true
	}
	return false
}

func (conv *Conv1D) PlainForwardCircuit(
	input []complex128, filters [][]complex128, sts *CellCnnSettings,
) []complex128 {
	slots := len(input)
	res := make([]complex128, slots)

	for i, filter := range filters {
		tmp := utils.SliceMult(input, filter)
		tmp = utils.SliceInnerSum(tmp, sts.Nmakers*sts.Ncells, 1)
		tmp = utils.SliceRotation(tmp, -i)
		res = utils.SliceAdd(res, tmp, false)
	}
	return res
}

func (conv *Conv1D) PlainBackwardCircuit(
	input []complex128, err0 []complex128, sts *CellCnnSettings,
) [][]complex128 {
	// get the err for each filter
	extErrSlice := make([][]complex128, sts.Nfilters)
	pf := complex(1.0/float64(sts.Ncells), 0)
	for i := 0; i < sts.Nfilters; i++ {
		erri := pf * err0[i*sts.Nclasses]
		extErrSlice[i] = utils.SliceFill(len(input), utils.NewSlice(0, sts.Ncells*sts.Nmakers-1, 1), erri)
	}

	// input transpose
	rowPacked := true
	inputT := utils.SliceTranspose(input, sts.Nmakers, sts.Ncells, rowPacked)

	dwSlice := make([][]complex128, sts.Nfilters)
	for i, extErr := range extErrSlice {

		// mult the extend err with the transpose input
		dwSlice[i] = utils.SliceMult(inputT, extErr)

		// innerSum to get the result in the correct place, valid at: (0~m-1)*ncells
		batch := sts.Ncells
		n := sts.Nmakers
		dwSlice[i] = utils.SliceInnerSum(dwSlice[i], batch, n)
		dwSlice[i] = utils.SliceCollect(dwSlice[i], utils.NewSlice(0, (sts.Nmakers-1)*sts.Ncells, sts.Ncells))
		dwSlice[i] = utils.SliceReplicate(dwSlice[i], sts.Nmakers, sts.Ncells)
	}

	return dwSlice
}
