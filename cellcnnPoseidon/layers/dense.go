package layers

import (
	"math"

	"github.com/ldsec/cellCNN/cellcnnPoseidon/approx/leastsquares"
	"github.com/ldsec/cellCNN/cellcnnPoseidon/utils"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
)

// Dense output layer for classification (nclasses > 1)
type Dense struct {
	weights   *ckks.Ciphertext   // column packed weights
	lastInput *ckks.Ciphertext   // nsamples x nfilters
	u         *ckks.Ciphertext   // output before activation, with garbage slots
	diagM     *ckks.PtDiagMatrix // for backward transpose operation, pre-computed in init rotation inds
	encoder   ckks.Encoder
	gradient  *ckks.Ciphertext
	momentum  float64
	vt        *ckks.Ciphertext
}

// NewDense constructor
func NewDense(weights *ckks.Ciphertext, momentum float64) *Dense {
	return &Dense{
		weights:  weights,
		momentum: momentum,
	}
}

// GetWeightsBinary  weights as byte
func (dense *Dense) GetWeightsBinary() []byte {
	data, err := dense.weights.MarshalBinary()
	if err != nil {
		panic("fail to GetWeightsBinary dense weights")
	}
	return data
}

// LoadWeightsBinary update the weights by data
func (dense *Dense) LoadWeightsBinary(data []byte) {
	nw := new(ckks.Ciphertext)
	if err := nw.UnmarshalBinary(data); err != nil {
		panic("fail to LoadWeightsBinary conv filter weights")
	}
	dense.weights = nw
}

// WithWeights debug use only
func (dense *Dense) WithWeights(weights *ckks.Ciphertext) {
	dense.weights = weights
}

// WithLastInput debug use only
func (dense *Dense) WithLastInput(input *ckks.Ciphertext) {
	dense.lastInput = input
}

// WithEncoder set the encoder for denses
func (dense *Dense) WithEncoder(encoder ckks.Encoder) {
	dense.encoder = encoder
}

// WithDiagM set the diagM for linear transformation in backward matrix transpose
func (dense *Dense) WithDiagM(diagM *ckks.PtDiagMatrix) {
	dense.diagM = diagM
}

// GetWeights return the weights ct
func (dense *Dense) GetWeights() *ckks.Ciphertext {
	return dense.weights
}

// FirstMomentum check if has first momentum
func (dense *Dense) FirstMomentum() bool {
	return dense.vt == nil
}

// UpdateMomentum update the momentum for next iteration
func (dense *Dense) UpdateMomentum(grad *ckks.Ciphertext) {
	dense.vt = grad
}

// InitRotationInds init the rotation indices
func (dense *Dense) InitRotationInds(sts *utils.CellCnnSettings, kgen ckks.KeyGenerator,
	params ckks.Parameters, encoder ckks.Encoder, maxM1N2Ratio float64,
) []int {
	nfilters := sts.Nfilters
	nclasses := sts.Nclasses

	// Dense Forward
	// 1. for replicate the input mutiple times
	Frep := params.RotationsForInnerSumLog(-sts.Nfilters, sts.Nclasses)
	// 2. for input weights matrix mult
	Fmult := params.RotationsForInnerSumLog(1, nfilters)
	Finds := append(Frep, Fmult...)

	// Dense Backward
	rot1 := params.RotationsForInnerSumLog(-1, sts.Nfilters)
	rot2 := params.RotationsForInnerSumLog(-sts.Nfilters, sts.Nclasses)
	rot3 := params.RotationsForInnerSumLog(-sts.Nclasses, sts.Nfilters)
	rotAll := append(rot1, rot2...)
	rotAll = append(rotAll, rot3...)

	// 	transpose
	inRowPacked := false
	ouRowPacked := false
	colsMatrix := utils.GenTransposeMatrix(params.Slots(), nfilters, nclasses, inRowPacked, ouRowPacked)
	transposeVec := utils.GenTransposeMap(colsMatrix)
	diagM := encoder.EncodeDiagMatrixBSGSAtLvl(params.MaxLevel(), transposeVec, params.Scale(), maxM1N2Ratio, params.LogSlots())

	dense.diagM = diagM
	Btranspose := params.RotationsForDiagMatrixMult(diagM)
	Binds := append(rotAll, Btranspose...)

	return append(Finds, Binds...)
}

// Forward forward only one sample, using newWeights if not nil
// return the prediction as a ct, the garbage slots are not masked,
// valid slots are at nfilters*(0~nclasses-1)
func (dense *Dense) Forward(input *ckks.Ciphertext, newWeights *ckks.Ciphertext, sts *utils.CellCnnSettings, eval ckks.Evaluator, encoder ckks.Encoder, params ckks.Parameters) *ckks.Ciphertext {

	//fmt.Printf("#### Dense forward Level Tracing ####\n")
	//fmt.Printf("p1 weights: %v\n", utils.PrintCipherLevel(dense.weights, params))
	//fmt.Printf("p2 input: %v\n", utils.PrintCipherLevel(input, params))

	if newWeights != nil {
		dense.weights = newWeights
	}

	dense.lastInput = input.CopyNew()

	var output *ckks.Ciphertext
	inputRep := input.CopyNew()

	// 1. replicate the input by Nclasses times
	eval.InnerSumLog(inputRep, -sts.Nfilters, sts.Nclasses, inputRep)

	// 2. mul the weights and the input representation
	inputRep = eval.MulRelinNew(inputRep, dense.weights)

	if err := eval.Rescale(inputRep, params.Scale(), inputRep); err != nil {
		panic("fail to rescale, dense")
	}

	// 3. innerSum to get the pred for class theta in the theta * nfilters place
	eval.InnerSumLog(inputRep, 1, sts.Nfilters, inputRep)

	output = inputRep

	// record the last ouput before activation for backward use.
	dense.u = output.CopyNew()
	// //fmt.Printf("Dense.u" + utils.PrintCipherLevel(dense.u, params))

	// 4. apply least square approximation to compute the sigmoid
	cfs, err := leastsquares.GetCoefficients(sts.Degree, sts.Interval)
	if err != nil {
		panic("something is wrong in dense least square approximation")
	}

	coeffs := make([]complex128, sts.Degree+1)
	coeffs[0] = complex(cfs[0], 0)

	for i := 1; i < len(coeffs); i++ {
		coeffs[i] = complex(cfs[i]/math.Pow(sts.Interval, float64(i)), 0)
	}

	poly := ckks.NewPoly(coeffs)

	output, err = eval.EvaluatePoly(output, poly, params.Scale())
	if err != nil {
		panic("something is wrong in dense least square approximation")
	}

	//fmt.Printf("p3 output: %v\n", utils.PrintCipherLevel(output, params))

	return output
}

// Backward compute the gradient
// inErr do not need to mask garbage slots
// sk used for dummy bootstrapping
// return the err to conv1d, and pure gradient
// for scaled one, call GetGradient
// for momentum one, call ComputeGradientWithMomentumAndLr
func (dense *Dense) Backward(inErr *ckks.Ciphertext, sts *utils.CellCnnSettings, params ckks.Parameters, eval ckks.Evaluator, encoder ckks.Encoder, sk *rlwe.SecretKey, lr float64) (*ckks.Ciphertext, *ckks.Ciphertext) {
	//fmt.Printf("#### Dense backward Level Tracing ####\n")
	//fmt.Printf("p1 InErr: %v\n", utils.PrintCipherLevel(inErr, params))
	//fmt.Printf("p2 Dense.u: %v\n", utils.PrintCipherLevel(dense.u, params))
	// 1. compute the derivative of the activation function
	cf, err := leastsquares.GetCoefficients(sts.Degree, sts.Interval)

	// scale coefficients based on the interval, first coeff will be gone when derivative...
	coeffs := make([]complex128, sts.Degree)
	coeffs[0] = complex(cf[1]/math.Pow(sts.Interval, float64(1)), 0)
	for i := 1; i < len(coeffs); i++ {
		coeffs[i] = complex((cf[i+1]/math.Pow(sts.Interval, float64(i+1)))*float64(i+1), 0)
	}

	poly := ckks.NewPoly(coeffs)

	dActv, err := eval.EvaluatePoly(dense.u, poly, params.Scale())
	if err != nil {
		panic("something is wrong in dense least square approximation")
	}
	//fmt.Printf("p3 dActv(Dense.u): %v\n", utils.PrintCipherLevel(dActv, params))

	// 2. mult the derivative and the inErr
	mErr := eval.MulRelinNew(dActv, inErr)
	if err := eval.Rescale(mErr, params.Scale(), mErr); err != nil {
		panic("fail to rescale, dense backward")
	}
	//fmt.Printf("p4 dActv(Dense.u) * InErr: %v\n", utils.PrintCipherLevel(mErr, params))

	// 3. compute the loss by mult last input.T (k*1) with err (1*theta)
	// because the mErr of dense (u or active(u)) already distributed in 0, k, 2k .. theta k place
	// here we directly right rotate (1~k-1) times and add together

	// mask the output slots at (nfilter*0~nclasses-1)
	maskOutput := utils.GenSliceWithOneAt(params.Slots(), utils.NewSlice(0, sts.Nfilters*(sts.Nclasses-1), sts.Nfilters))
	plainMask := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), maskOutput, params.LogSlots())
	eval.MulRelin(mErr, plainMask, mErr)
	if err := eval.Rescale(mErr, params.Scale(), mErr); err != nil {
		panic("fail to rescale in dsbackward mask")
	}
	//fmt.Printf("p5 mask garb p4: %v\n", utils.PrintCipherLevel(mErr, params))

	// dummy bootstrapping to recover the cipheretext to the inital level.
	if sk != nil {
		mErr = utils.DummyBootstrapping(mErr, params, sk)
	}
	//fmt.Printf("p6 bootstrap(p5): %v\n", utils.PrintCipherLevel(mErr, params))

	repErr := mErr.CopyNew()
	eval.InnerSumLog(repErr, -1, sts.Nfilters, repErr)

	// 4. replicate the last input theta times
	repInput := dense.lastInput.CopyNew()
	eval.InnerSumLog(repInput, -sts.Nfilters, sts.Nclasses, repInput)
	//fmt.Printf("p7 dense last input: %v\n", utils.PrintCipherLevel(repInput, params))

	// 5. mult the err and the input get the derivative for the dense weights
	dense.gradient = eval.MulRelinNew(repInput, repErr)
	if err := eval.Rescale(dense.gradient, params.Scale(), dense.gradient); err != nil {
		panic("fail to rescale, dense backward dw")
	}
	//fmt.Printf("p8 gradient: p6 mult p7: %v\n", utils.PrintCipherLevel(dense.gradient, params))

	// keep pure gradient
	pureGradient := dense.gradient.CopyNew()

	// compute scaled gradient
	if lr != 0 {
		dense.ComputeScaledGradient(dense.gradient, sts, params, eval, encoder, lr)
	}

	//  prepare the transpose weights for later computation
	weightsT := eval.LinearTransform(dense.weights, dense.diagM)[0]
	if err := eval.Rescale(weightsT, params.Scale(), weightsT); err != nil {
		panic("fail to rescale, dense backward weightsT")
	}
	//fmt.Printf("p9 weights transpose: %v\n", utils.PrintCipherLevel(weightsT, params))

	// 8. compute the err to next layer, mask garbage slots and collect the valid slots to left most
	mErrCollect := utils.MaskAndCollectToLeft(mErr, params, encoder, eval, 0, sts.Nfilters, sts.Nclasses)
	// Please use this function instead if nfilters > nclasses
	// mErrCollect := utils.MaskAndCollectToLeftFast(mErr, params, encoder, eval, 0, sts.Nfilters, sts.Nclasses, true, true)

	// replicate the err k times and mult with weightsT
	eval.InnerSumLog(mErrCollect, -sts.Nclasses, sts.Nfilters, mErrCollect)

	outErr := eval.MulRelinNew(mErrCollect, weightsT)
	if err := eval.Rescale(outErr, params.Scale(), outErr); err != nil {
		panic("fail to rescale, dense backward outErr")
	}

	// 2. innerSum to get the err in nclasses * (0~k-1) slots, with garbage
	eval.InnerSumLog(outErr, 1, sts.Nclasses, outErr)
	//fmt.Printf("p10 weight.T mult mErr: %v\n", utils.PrintCipherLevel(outErr, params))

	return outErr, pureGradient
}

// ComputeScaledGradient compute the scaled gradients: scaled = pure * lr
func (dense *Dense) ComputeScaledGradient(
	gradient *ckks.Ciphertext,
	sts *utils.CellCnnSettings, params ckks.Parameters,
	eval ckks.Evaluator, encoder ckks.Encoder, lr float64,
) {
	eval.MultByConst(gradient, lr, gradient)
	if err := eval.Rescale(gradient, params.Scale(), gradient); err != nil {
		panic("backward dense: fail to rescale, update")
	}
	dense.gradient = gradient
}

// ComputeScaledGradientWithMomentum compute the scaled with momentum:
// g_{new} = g_{old} + momentum
func (dense *Dense) ComputeScaledGradientWithMomentum(
	gradient *ckks.Ciphertext,
	sts *utils.CellCnnSettings, params ckks.Parameters,
	eval ckks.Evaluator, encoder ckks.Encoder, momentum float64,
) *ckks.Ciphertext {
	if momentum > 0 {
		if dense.vt == nil {
			dense.vt = gradient.CopyNew()
		} else {
			dense.vt = eval.MultByConstNew(dense.vt, momentum)
			if err := eval.Rescale(dense.vt, params.Scale(), dense.vt); err != nil {
				panic("backward: fail to rescale, conv.vt[i]")
			}
			dense.vt = eval.AddNew(dense.vt, gradient)
		}
		dense.gradient = dense.vt.CopyNew()
	} else {
		dense.gradient = gradient
	}
	return dense.gradient
}

// GetGradient get the gradient of the model, may be pure / scaled / momentumed
func (dense *Dense) GetGradient() *ckks.Ciphertext {
	return dense.gradient.CopyNew()
}

// GetGradientBinary gradients as byte
func (dense *Dense) GetGradientBinary() []byte {
	data, err := dense.gradient.MarshalBinary()
	if err != nil {
		panic("fail to GetGradientBinary dense weights")
	}
	return data
}

// UpdateWithGradients subtract g from self.weights: w = w - g
func (dense *Dense) UpdateWithGradients(g *ckks.Ciphertext, eval ckks.Evaluator) {
	eval.Sub(dense.weights, g, dense.weights)
}

// PlainForwardCircuit debug use only, forward the input to plaintext circuit
func (dense *Dense) PlainForwardCircuit(weights []complex128, input []complex128, sts *utils.CellCnnSettings) ([]complex128, []complex128) {

	inputRep := utils.SliceReplicate(input, sts.Nfilters, sts.Nclasses)

	multOut := utils.SliceMult(inputRep, weights)

	innerSumOut := utils.SliceInnerSum(multOut, sts.Nfilters, sts.Nclasses)
	// valid slots
	n := sts.Nfilters * sts.Nclasses
	// get coeffs
	cf, err := leastsquares.GetCoefficients(sts.Degree, sts.Interval)
	if err != nil {
		panic("fail to get coeffs")
	}
	coeffs := make([]complex128, sts.Degree+1)
	coeffs[0] = complex(cf[0], 0)

	for i := 1; i < len(coeffs); i++ {
		coeffs[i] = complex(cf[i]/math.Pow(sts.Interval*float64(sts.Ncells), float64(i)), 0)
	}

	polyOut := utils.SliceEvaluatePoly(innerSumOut, coeffs, n)
	return polyOut, innerSumOut
}

// PlainBackwardCircuit debug use only, backward one sample
func (dense *Dense) PlainBackwardCircuit(
	weights []complex128, input []complex128, u []complex128, err0 []complex128, sts *utils.CellCnnSettings,
) ([]complex128, []complex128) {
	cf, err := leastsquares.GetCoefficients(sts.Degree, sts.Interval)
	if err != nil {
		panic("fail to get coeffs")
	}
	// valid slots
	n := sts.Nfilters * sts.Nclasses
	// scale coefficients based on the interval
	coeffs := make([]complex128, sts.Degree)
	//first coeff will be gone when derivative...
	coeffs[0] = complex(cf[1]/math.Pow(sts.Interval, float64(1)), 0)
	for i := 1; i < len(coeffs); i++ {
		coeffs[i] = complex((cf[i+1]/math.Pow(sts.Interval*float64(sts.Ncells), float64(i+1)))*float64(i+1), 0)
	}
	dActv := utils.SliceEvaluatePoly(u, coeffs, n)
	// err 0 mult derivative of the activation function
	mErr := utils.SliceMult(err0, dActv)

	// mult the input (k*1) and the err (1*theta)
	extErr := utils.SliceExtend(
		mErr, utils.NewSlice(0, sts.Nfilters*(sts.Nclasses-1), sts.Nfilters), sts.Nfilters,
	)

	extInput := utils.SliceReplicate(input, sts.Nfilters, sts.Nclasses)

	dw := utils.SliceMult(extErr, extInput) //dw := ...

	wt := utils.SliceTranspose(weights, sts.Nclasses, sts.Nfilters, false)

	isNegative := true

	// this is the updated dense weights, without multiply with learning rate
	_ = utils.SliceAdd(weights, dw, isNegative)

	cmErr := utils.SliceCollect(mErr, utils.NewSlice(0, sts.Nfilters*(sts.Nclasses-1), sts.Nfilters))

	cmErrRep := utils.SliceReplicate(cmErr, sts.Nclasses, sts.Nfilters)

	outErr := utils.SliceMult(cmErrRep, wt)

	innerSumOutErr := utils.SliceInnerSum(outErr, sts.Nclasses, sts.Nfilters)

	// output err with size (1 * Nfilters)
	// the valid results are in slots Nclasses * (0 ~ Nfilters-1)

	return dw, innerSumOutErr
}
