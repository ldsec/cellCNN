package layers

import (
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
	Brep := make([]int, 0)
	for i := 0; i < sts.Nfilters; i++ {
		extInds := utils.GenExtentionInds(i*sts.Nclasses, utils.NewSlice(0, sts.Ncells*sts.Nmakers-1, 1))
		Brep = append(Brep, extInds...)
	}
	// 4. left collect gradient
	Bcol := make([]int, sts.Nmakers)
	for i := 0; i < sts.Nmakers; i++ {
		Bcol[i] = sts.Ncells*i - i
	}
	// 5. replicate gradient
	Brepg := utils.NegativeSlice(utils.NewSlice(0, (sts.Ncells-1)*sts.Nmakers, sts.Nmakers))
	Binds := append(Brep, Bcol...)
	Binds = append(Binds, Brepg...)

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

	//  2.2 rotate to extend one slots to ncells*nmakers, need to generate new rotation keys
	extErrSlice := make([]*ckks.Ciphertext, sts.Nfilters)
	for i, mErr := range maskedErrSlice {
		extInds := utils.GenExtentionInds(i*sts.Nclasses, utils.NewSlice(0, sts.Ncells*sts.Nmakers-1, 1))
		rotMap := evaluator.RotateHoisted(mErr, extInds)
		extErrSlice[i] = utils.AddHoistedMap(rotMap, evaluator)
	}

	// 3. mult the extended err with the transposed last input
	dwSlice := make([]*ckks.Ciphertext, sts.Nfilters)
	inputT := conv.TransposeInput(sts, conv.lastInput, params)

	batch := 1
	n := sts.Ncells
	for i, extErr := range extErrSlice {
		dwSlice[i] = evaluator.MulRelinNew(inputT, extErr)
		if err := evaluator.Rescale(dwSlice[i], params.Scale(), dwSlice[i]); err != nil {
			panic("fail to rescale, conv backward, inputT mult extErr")
		}
		// if i == 0 {
		// 	fmt.Printf("inErr - 2 " + utils.PrintCipherLevel(dwSlice[i], params))
		// }

		// 4. innerSum to get the result in the correct place, valid at: (0~m-1)*n
		evaluator.InnerSum(dwSlice[i], batch, n, dwSlice[i])

		// 5. mask & rotate to sum the result in the left most slots
		// require new rotation ids
		dwSlice[i] = utils.MaskAndCollectToLeft(dwSlice[i], params, encoder, evaluator, 0, sts.Ncells, sts.Nmakers)

		// 6. replicate ncells times
		extInds := utils.NegativeSlice(utils.NewSlice(0, (sts.Ncells-1)*sts.Nmakers, sts.Nmakers))
		extMap := evaluator.RotateHoisted(dwSlice[i], extInds)
		dwSlice[i] = utils.AddHoistedMap(extMap, evaluator)
	}

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
