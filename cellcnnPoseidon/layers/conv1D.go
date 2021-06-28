package layers

import (
	"fmt"

	"github.com/ldsec/cellCNN/cellcnnPoseidon/utils"
	"github.com/ldsec/lattigo/v2/ckks"
)

// Conv1D conduct 1D convolution and pooling together
type Conv1D struct {
	filters   []*ckks.Ciphertext // len(filters) = Nfilters
	lastInput *ckks.Plaintext    // packed ncells after encoding

	encoder  ckks.Encoder
	gradient []*ckks.Ciphertext
	momentum float64
	vt       []*ckks.Ciphertext // momentum
}

// NewConv1D constructor
func NewConv1D(filters []*ckks.Ciphertext, isMomentum float64) *Conv1D {
	return &Conv1D{
		filters:  filters,
		momentum: isMomentum,
		vt:       make([]*ckks.Ciphertext, len(filters)),
		gradient: make([]*ckks.Ciphertext, len(filters)),
	}
}

// GetWeightsBinary return the encrypted weights as bytes
func (conv *Conv1D) GetWeightsBinary() [][]byte {
	var err error
	Nfilters := len(conv.filters)
	data := make([][]byte, Nfilters)
	for i := 0; i < Nfilters; i++ {
		data[i], err = conv.filters[i].MarshalBinary()
		if err != nil {
			panic("fail to GetWeightsBinary conv filter weights")
		}
	}
	return data
}

// LoadWeightsBinary retreive the new weights
func (conv *Conv1D) LoadWeightsBinary(data [][]byte) {
	conv.filters = make([]*ckks.Ciphertext, len(data))
	for i, each := range data {
		conv.filters[i] = new(ckks.Ciphertext)
		if err := conv.filters[i].UnmarshalBinary(each); err != nil {
			panic("fail to LoadWeightsBinary conv filter weights")
		}
	}
}

// WithWeights for debug use only
func (conv *Conv1D) WithWeights(filters []*ckks.Ciphertext) {
	conv.filters = filters
}

// WithLastInput for debug use only
func (conv *Conv1D) WithLastInput(input *ckks.Plaintext) {
	conv.lastInput = input
}

// WithEncoder set the encoder to use
func (conv *Conv1D) WithEncoder(encoder ckks.Encoder) {
	conv.encoder = encoder
}

// UpdateWithGradients update the weights by:
// self = self - g
func (conv *Conv1D) UpdateWithGradients(g []*ckks.Ciphertext, eval ckks.Evaluator) {
	for i := range conv.filters {
		eval.Sub(conv.filters[i], g[i], conv.filters[i])
	}
}

// GetWeights return the weights as ciphertext slice
func (conv *Conv1D) GetWeights() []*ckks.Ciphertext {
	return conv.filters
}

// FirstMomentum check if has first momentum
func (conv *Conv1D) FirstMomentum() bool {
	return conv.vt[0] == nil
}

// UpdateMomentum save the grad as new momentum for next iteration
func (conv *Conv1D) UpdateMomentum(grad []*ckks.Ciphertext) {
	conv.vt = grad
}

// InitRotationInds init the rotation keys
func (conv *Conv1D) InitRotationInds(sts *utils.CellCnnSettings, params ckks.Parameters) []int {
	nmakers := sts.Nmakers
	nfilters := sts.Nfilters
	ncells := sts.Ncells

	// Conv1D Forward
	// 1. for input weights matrix mult
	Fmult := params.RotationsForInnerSumLog(1, nmakers*ncells)
	// 2. for rotation the result to left most slots
	Fshift := utils.NegativeSlice(utils.NewSlice(1, nfilters-1, 1))
	Finds := append(Fmult, Fshift...)

	//Conv1D Backward
	// 3. for replicate the err for each filter
	Brep1 := utils.NewSlice(0, (sts.Nfilters-1)*sts.Nclasses, sts.Nclasses)
	Brep2 := params.RotationsForInnerSumLog(-1, sts.Ncells*sts.Nmakers)
	Brep := append(Brep1, Brep2...)

	// 4. left collect gradient
	Bcol := make([]int, sts.Nmakers)
	for i := 0; i < sts.Nmakers; i++ {
		Bcol[i] = sts.Ncells*i - i
	}
	// 5. replicate gradient
	Brepg := params.RotationsForInnerSumLog(-sts.Nmakers, sts.Ncells)
	B0 := params.RotationsForInnerSumLog(1, sts.Ncells)
	Bcollect := params.RotationsForInnerSumLog(sts.Ncells-1, sts.Nmakers)

	Binds := append(Brep, Bcol...)
	Binds = append(Binds, Brepg...)
	Binds = append(Binds, B0...)
	Binds = append(Binds, Bcollect...)

	Inds := append(Finds, Binds...)

	return Inds
}

// TransposeInput transpose the input for backward.
func (conv *Conv1D) TransposeInput(sts *utils.CellCnnSettings, input *ckks.Plaintext, params ckks.Parameters) *ckks.Plaintext {
	slice := conv.encoder.Decode(input, params.LogSlots())
	rowPacked := true
	sliceT := utils.SliceTranspose(slice, sts.Nmakers, sts.Ncells, rowPacked)
	plaintextT := conv.encoder.EncodeNTTAtLvlNew(params.MaxLevel(), sliceT, params.LogSlots())
	return plaintextT
}

// Forward computes a forward pass of the Conv1D layer, using newFilters if not nil
// activation function is identity
func (conv *Conv1D) Forward(input *ckks.Plaintext, newFilters []*ckks.Ciphertext, sts *utils.CellCnnSettings, eval ckks.Evaluator, params ckks.Parameters) *ckks.Ciphertext {

	//fmt.Printf("#### Conv1d forward Level Tracing ####\n")
	////fmt.Printf("p1 weights: %v\n", utils.PrintCipherLevel(conv.filters[0], params))

	if newFilters != nil {
		conv.filters = newFilters
	}

	conv.lastInput = input

	actvs := make([]*ckks.Ciphertext, len(conv.filters)) // activations for each filter

	// mask for the first slot as filer response on batch cells
	leftMostMask := utils.GenSliceWithAnyAt(params.Slots(), []int{0}, 1.0/float64(sts.Ncells))
	poolMask := conv.encoder.EncodeNTTAtLvlNew(params.MaxLevel(), leftMostMask, params.LogSlots())

	var output *ckks.Ciphertext
	batch := 1                    // only the left most
	n := sts.Nmakers * sts.Ncells // num of slots to add together

	// loop over all filters
	for i, filter := range conv.filters {

		// 1. multiply the filters and the input
		actvs[i] = eval.MulRelinNew(filter, input)

		eval.Rescale(actvs[i], params.Scale(), actvs[i])

		// 2. innerSum to get the pooling result (before avg) in the left most slot
		eval.InnerSumLog(actvs[i], batch, n, actvs[i])

		// 3. mask the ouput to keep only the left most element
		eval.MulRelin(actvs[i], poolMask, actvs[i])
		eval.Rescale(actvs[i], params.Scale(), actvs[i])

		// 4. rotate the result to i-th place and add together
		if i == 0 {
			output = actvs[i]
		} else {
			eval.Rotate(actvs[i], -i, actvs[i])
			eval.Add(output, actvs[i], output)
		}
	}

	//fmt.Printf("p2 output: %v\n", utils.PrintCipherLevel(output, params))

	return output
}

// Backward compute the gradient
// return the scaled, no-momentum, un-replicated gradient
// for momentum one, call ComputeScaledGradientWithMomentum
func (conv *Conv1D) Backward(
	inErr *ckks.Ciphertext, sts *utils.CellCnnSettings, params ckks.Parameters,
	evaluator ckks.Evaluator, encoder ckks.Encoder, lr float64,
) []*ckks.Ciphertext {

	//fmt.Printf("#### Conv1d backward Level Tracing ####\n")
	//fmt.Printf("p1 InErr: %v\n", utils.PrintCipherLevel(inErr, params))
	// 1. mult the dActv with the income err, currently dActv = 1, skip this part

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
	}

	//  2.2 rotate to extend one slots to ncells*nmakers, need to generate new rotation keys
	extErrSlice := make([]*ckks.Ciphertext, sts.Nfilters)
	for i, mErr := range maskedErrSlice {
		// left rotate once and right replicate
		leftMostTmp := evaluator.RotateNew(mErr, i*sts.Nclasses)
		evaluator.InnerSumLog(leftMostTmp, -1, sts.Ncells*sts.Nmakers, leftMostTmp)
		extErrSlice[i] = leftMostTmp
	}
	//fmt.Printf("p2 upsamling the inErr: %v\n", utils.PrintCipherLevel(maskedErrSlice[0], params))

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

		// 4. innerSum to get the result in the correct place, valid at: (0~m-1)*n
		evaluator.InnerSumLog(dwSlice[i], batch, n, dwSlice[i])

		// 5. mask & rotate to sum the result in the left most slots
		dwSlice[i] = utils.MaskAndCollectToLeftFast(dwSlice[i], params, encoder, evaluator, 0, sts.Ncells, sts.Nmakers, false, false)
	}

	//fmt.Printf("p3 gradient: %v\n", utils.PrintCipherLevel(dwSlice[0], params))

	// pure and no momentum gradient
	conv.gradient = utils.CopyCiphertextSlice(dwSlice)

	// compute the scaled gradient
	if lr != 0 {
		conv.ComputeScaledGradient(conv.gradient, sts, params, evaluator, encoder, lr)
	}

	return dwSlice
}

// ComputeScaledGradient compute the scaled gradients: scaled_grad = pure_grad * lr
// results in conv.gradients
func (conv *Conv1D) ComputeScaledGradient(
	gradients []*ckks.Ciphertext,
	sts *utils.CellCnnSettings, params ckks.Parameters,
	eval ckks.Evaluator, encoder ckks.Encoder, lr float64,
) {
	// create mask with scale
	maskInds := utils.NewSlice(0, sts.Nmakers, 1)
	mask := utils.GenSliceWithAnyAt(params.Slots(), maskInds, lr)
	maskPlain := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), mask, params.LogSlots())
	for i := range gradients {
		// 1. mask and scale
		eval.MulRelin(gradients[i], maskPlain, gradients[i])
		if err := eval.Rescale(gradients[i], params.Scale(), gradients[i]); err != nil {
			panic("fail to rescale, conv.gradient[i]")
		}
		// 2. replicate ncell times
		eval.InnerSumLog(gradients[i], -sts.Nmakers, sts.Ncells, gradients[i])
	}
	conv.gradient = gradients
}

// ComputeScaledGradientWithMomentum add the momentum to the scaled gradients
func (conv *Conv1D) ComputeScaledGradientWithMomentum(
	gradients []*ckks.Ciphertext,
	sts *utils.CellCnnSettings, params ckks.Parameters,
	eval ckks.Evaluator, encoder ckks.Encoder, momentum float64,
) []*ckks.Ciphertext {
	for i := range gradients {
		if momentum > 0 {
			fmt.Printf("len vt: %v, grad: %v, self grad: %v\n", len(conv.vt), len(gradients), len(conv.gradient))
			if conv.vt[i] == nil {
				conv.vt[i] = gradients[i].CopyNew()
			} else {
				conv.vt[i] = eval.MultByConstNew(conv.vt[i], momentum)
				if err := eval.Rescale(conv.vt[i], params.Scale(), conv.vt[i]); err != nil {
					panic("backward: fail to rescale, conv.vt[i]")
				}
				conv.vt[i] = eval.AddNew(conv.vt[i], gradients[i])
			}
			conv.gradient[i] = conv.vt[i].CopyNew()
		} else {
			// else use only gradient
			conv.gradient[i] = gradients[i]
		}
	}
	return conv.gradient
}

// GetGradient return the gradients
func (conv *Conv1D) GetGradient() []*ckks.Ciphertext {
	return utils.CopyCiphertextSlice(conv.gradient)
}

// GetGradientBinary return the gradients represented as byte
func (conv *Conv1D) GetGradientBinary() [][]byte {
	var err error
	Nfilters := len(conv.filters)
	data := make([][]byte, Nfilters)
	for i := 0; i < Nfilters; i++ {
		data[i], err = conv.gradient[i].MarshalBinary()
		if err != nil {
			panic("fail to GetGradientBinary conv filter weights")
		}
	}
	return data
}

// PlainForwardCircuit debug use only, forward a sample to plaintext circuit
func (conv *Conv1D) PlainForwardCircuit(input []complex128, filters [][]complex128, sts *utils.CellCnnSettings) []complex128 {
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

// PlainBackwardCircuit debug use only, backward an error to plaintext circuit
func (conv *Conv1D) PlainBackwardCircuit(
	input []complex128, err0 []complex128, sts *utils.CellCnnSettings,
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
