package layers

import (
	"fmt"
	"math"

	"github.com/ldsec/cellCNN/semester_project_shufan/approx/leastsquares"
	"github.com/ldsec/cellCNN/semester_project_shufan/utils"
	"github.com/ldsec/lattigo/v2/ckks"
)

// requires rotation keys for:
//	slots - [1, Nclasses-1] * nfilters
//  [1, nfilters-1]
// 	[1, Nclasses-1] * (nfilters - 1)

// Dense output layer for classification (nclasses > 1)
type Dense struct {
	weights    *ckks.Ciphertext // column packed weights
	lastInput  *ckks.Ciphertext // nsamples x nfilters
	u          *ckks.Ciphertext
	diagM      *ckks.PtDiagMatrix // for backward transpose operation
	encoder    ckks.Encoder
	gradient   *ckks.Ciphertext
	isMomentum bool
	vt         *ckks.Ciphertext
	// Activation  func(float64) float64
	// dActivation func(float64) float64
}

// NewDense constructor
func NewDense(weights *ckks.Ciphertext) *Dense {
	return &Dense{
		weights:    weights,
		isMomentum: false,
	}
}

// ################# Debug Functions ####################
func (dense *Dense) WithWeights(weights *ckks.Ciphertext) {
	dense.weights = weights
}

func (dense *Dense) WithLastInput(input *ckks.Ciphertext) {
	dense.lastInput = input
}

func (dense *Dense) WithEncoder(encoder ckks.Encoder) {
	dense.encoder = encoder
}

func (dense *Dense) GetGradient() *ckks.Ciphertext {
	return dense.gradient
}

func (dense *Dense) GetWeights() *ckks.Ciphertext {
	return dense.weights
}

func (dense *Dense) SetMomentum() {
	dense.isMomentum = true
}

func (dense *Dense) InitRotationInds(sts *CellCnnSettings, kgen ckks.KeyGenerator,
	params *ckks.Parameters, encoder ckks.Encoder, maxM1N2Ratio float64,
) []int {
	// maxM1N2Ratio = 8.0
	// nmakers := sts.Nmakers
	nfilters := sts.Nfilters
	// ncells := sts.Ncells
	nclasses := sts.Nclasses

	// Dense Forward
	// 1. for replicate the input mutiple times
	Frep := make([]int, 0)
	for i := 1; i < nclasses; i++ {
		Frep = append(Frep, -i*nfilters)
	}
	// 2. for input weights matrix mult
	Fmult := kgen.GenRotationIndexesForInnerSum(1, nfilters)
	// 3. for collect the result into the left most slots
	// step := nfilters - 1
	// Fshift := utils.NewSlice(0, (nclasses-1)*step, step)
	Finds := append(Frep, Fmult...)
	// Finds = append(Finds, Fshift...)

	// Dense Backward

	// 	err mult input
	// 	rotation keys required: -1 ~ -(k-1)
	// 	rotation keys required: -k ~ -k*(theta-1)
	rot1 := utils.NegativeSlice(utils.NewSlice(1, sts.Nfilters-1, 1))
	rot2 := utils.NegativeSlice(utils.NewSlice(sts.Nfilters, sts.Nfilters*(sts.Nclasses-1), sts.Nfilters))
	rotAll := append(rot1, rot2...)

	// 	transpose
	inRowPacked := false
	ouRowPacked := false
	colsMatrix := utils.GenTransposeMatrix(params.Slots(), nfilters, nclasses, inRowPacked, ouRowPacked)
	transposeVec := utils.GenTransposeMap(colsMatrix)
	diagM := encoder.EncodeDiagMatrixAtLvl(params.MaxLevel(), transposeVec, params.Scale(), maxM1N2Ratio, params.LogSlots())
	dense.diagM = diagM
	Btranspose := kgen.GenRotationIndexesForDiagMatrix(diagM)
	Binds := append(rotAll, Btranspose...)

	return append(Finds, Binds...)
}

// Forward pass of the Dense_n layer, using newWeights if not nil
func (dense *Dense) Forward(
	input *ckks.Ciphertext,
	newWeights *ckks.Ciphertext,
	sts *CellCnnSettings,
	evaluator ckks.Evaluator,
	encoder ckks.Encoder,
	params *ckks.Parameters,
	maskMap map[int]*ckks.Plaintext,
) *ckks.Ciphertext {

	if newWeights != nil {
		dense.weights = newWeights
	}

	// records the last input for backward
	dense.lastInput = input

	var inputRep, tmp, output *ckks.Ciphertext
	inputRep = input

	// 1. replicate the input by Nclasses times
	for i := 1; i < sts.Nclasses; i++ {
		tmp = evaluator.RotateNew(input, -i*sts.Nfilters)
		evaluator.Add(inputRep, tmp, inputRep)
	}

	// 2. mul the weights and the input representation
	inputRep = evaluator.MulRelinNew(inputRep, dense.weights)

	if err := evaluator.Rescale(inputRep, params.Scale(), inputRep); err != nil {
		panic("fail to rescale, dense")
	}

	// 3. innerSum to get the pred for class theta in the theta * nfilters place
	evaluator.InnerSum(inputRep, 1, sts.Nfilters, inputRep)

	// 4. mask and left shift to store the result in left most Nclass slots.
	// // mask the output slots at (nfilter*0~nclasses-1)
	// maskOutput := utils.GenSliceWithOneAt(params.Slots(), utils.NewSlice(0, sts.Nfilters*(sts.Nclasses-1), sts.Nfilters))
	// plainMask := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), maskOutput, params.LogSlots())
	// evaluator.MulRelin(inputRep, plainMask, inputRep)
	// if err := evaluator.Rescale(inputRep, params.Scale(), inputRep); err != nil {
	// 	panic("fail to rescale in dsbackward mask")
	// }

	output = inputRep

	// record the last ouput before activation for backward use.
	dense.u = output.CopyNew().Ciphertext()
	// fmt.Printf("Dense.u" + utils.PrintCipherLevel(dense.u, params))

	// 5. apply least square approximation to compute the sigmoid
	cfs, err := leastsquares.GetCoefficients(sts.Degree, sts.Interval)
	fmt.Printf("Using approx coefficient: %v\n", cfs)
	if err != nil {
		panic("something is wrong in dense least square approximation")
	}

	// use "leastsquaresOptEco" as implemented in
	coeffs := make([]complex128, sts.Degree+1)
	coeffs[0] = complex(cfs[0], 0)

	for i := 1; i < len(coeffs); i++ {
		coeffs[i] = complex(cfs[i]/math.Pow(sts.Interval*float64(sts.Ncells), float64(i)), 0)
	}

	poly := ckks.NewPoly(coeffs)

	// u-2 level output
	output, err = evaluator.EvaluatePoly(output, poly, params.Scale())
	if err != nil {
		panic("something is wrong in dense least square approximation")
	}

	return output
}

func (dense *Dense) Backward(
	inErr *ckks.Ciphertext, sts *CellCnnSettings, params *ckks.Parameters,
	evaluator ckks.Evaluator, encoder ckks.Encoder, sk *ckks.SecretKey,
) *ckks.Ciphertext {
	// 1. compute the derivative of the activation function
	cf, err := leastsquares.GetCoefficients(sts.Degree, sts.Interval)

	// scale coefficients based on the interval
	coeffs := make([]complex128, sts.Degree)
	//first coeff will be gone when derivative...
	coeffs[0] = complex(cf[1]/math.Pow(sts.Interval, float64(1)), 0)
	for i := 1; i < len(coeffs); i++ {
		coeffs[i] = complex((cf[i+1]/math.Pow(sts.Interval*float64(sts.Ncells), float64(i+1)))*float64(i+1), 0)
	}

	poly := ckks.NewPoly(coeffs)

	// ------- u-2 level
	dActv, err := evaluator.EvaluatePoly(dense.u, poly, params.Scale())
	// fmt.Printf("u-2 " + utils.PrintCipherLevel(dActv, params))
	// fmt.Printf("inErr " + utils.PrintCipherLevel(inErr, params))
	if err != nil {
		panic("something is wrong in dense least square approximation")
	}

	// ------- b = min{u-3, inErr-1} level
	// 2. mult the derivative and the inErr
	mErr := evaluator.MulRelinNew(dActv, inErr)
	if err := evaluator.Rescale(mErr, params.Scale(), mErr); err != nil {
		panic("fail to rescale, dense backward")
	}
	// fmt.Printf("b = min{u-3, inErr-1} " + utils.PrintCipherLevel(mErr, params))
	if sk != nil {
		// fmt.Printf("Before re-encrypt: " + utils.PrintCipherLevel(mErr, params))
		ect := ckks.NewEncryptorFromSk(params, sk)
		dct := ckks.NewDecryptor(params, sk)
		cmplSlice := encoder.Decode(dct.DecryptNew(mErr), params.LogSlots())
		replain := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), cmplSlice, params.LogSlots())
		mErr = ect.EncryptNew(replain)
		// fmt.Printf("After re-encrypt: " + utils.PrintCipherLevel(mErr, params))
	}

	// 3. compute the loss by mult last input.T (k*1) with err (1*theta)
	// because the mErr of dense (u or active(u)) already distributed in 0, k, 2k .. theta k place
	// here we directly right rotate (1~k-1) times and add together
	// rotation keys required: -1 ~ -(k-1)

	// here we need mask garbage slots
	// mask the output slots at (nfilter*0~nclasses-1)
	maskOutput := utils.GenSliceWithOneAt(params.Slots(), utils.NewSlice(0, sts.Nfilters*(sts.Nclasses-1), sts.Nfilters))
	plainMask := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), maskOutput, params.LogSlots())

	// ------- b-1 level
	evaluator.MulRelin(mErr, plainMask, mErr)
	if err := evaluator.Rescale(mErr, params.Scale(), mErr); err != nil {
		panic("fail to rescale in dsbackward mask")
	}
	// fmt.Printf("b-1 " + utils.PrintCipherLevel(mErr, params))

	rotIndsErr := utils.NegativeSlice(utils.NewSlice(0, sts.Nfilters-1, 1))
	rotMapErr := evaluator.RotateHoisted(mErr, rotIndsErr)

	// sums up the rotated ciphertexts
	repErr := rotMapErr[0]
	for i := 1; i < sts.Nfilters; i++ {
		evaluator.Add(repErr, rotMapErr[-i], repErr)
	}

	// 4. replicate the last input theta times
	// rotation keys required: -k ~ -k*(theta-1)

	rotIndsInput := utils.NegativeSlice(utils.NewSlice(0, sts.Nfilters*(sts.Nclasses-1), sts.Nfilters))
	rotMapInput := evaluator.RotateHoisted(dense.lastInput, rotIndsInput)

	// sums up the rotated ciphertexts
	repInput := rotMapInput[0]
	for i := 1; i < sts.Nclasses-1; i++ {
		evaluator.Add(repInput, rotMapInput[-sts.Nfilters*i], repInput)
	}

	// 5. mult the err and the input get the derivative for the dense weights
	// ------- min{b-2, input-1} level
	repInput = evaluator.MultByConstNew(repInput, 0.5)
	dense.gradient = evaluator.MulRelinNew(repInput, repErr)
	if err := evaluator.Rescale(dense.gradient, params.Scale(), dense.gradient); err != nil {
		panic("fail to rescale, dense backward dw")
	}
	// fmt.Printf("min{b-2, input-1} " + utils.PrintCipherLevel(dense.gradient, params))

	// return dw

	// 6. mult the dw with the learning rate

	// 6.5 prepare the transpose weights for later computation
	// this should be conducted before the updates of w

	// ------- max-1 level
	weightsT := evaluator.LinearTransform(dense.weights, dense.diagM)[0]
	if err := evaluator.Rescale(weightsT, params.Scale(), weightsT); err != nil {
		panic("fail to rescale, dense backward weightsT")
	}
	// fmt.Printf("max-1 " + utils.PrintCipherLevel(weightsT, params))

	// 7. sub the dw from the weights
	// dense.weights = evaluator.SubNew(dense.weights, dw)

	// 8. compute the err which will be passed to next layer.
	// first rotate the input to get the results in the left most slots
	rotIndsCollect := utils.NewSlice(0, (sts.Nfilters-1)*(sts.Nclasses-1), sts.Nfilters-1)
	rotMapCollect := evaluator.RotateHoisted(mErr, rotIndsCollect)

	// mask and add together
	// mErrCollect := rotMapCollect[0]
	var mErrCollect *ckks.Ciphertext
	for i := 0; i < sts.Nclasses; i++ {
		mask := utils.GenSliceWithOneAt(params.Slots(), []int{i})
		maskPlain := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), mask, params.LogSlots())

		// ------- b = min{u-4, inErr-2} level
		evaluator.MulRelin(rotMapCollect[i*(sts.Nfilters-1)], maskPlain, rotMapCollect[i*(sts.Nfilters-1)])
		if err := evaluator.Rescale(rotMapCollect[i*(sts.Nfilters-1)], params.Scale(), rotMapCollect[i*(sts.Nfilters-1)]); err != nil {
			panic("fail to rescale, dense backward, err collect to left")
		}
		// if i == 0 {
		// 	fmt.Printf("min{u-4, inErr-2, b-2} " + utils.PrintCipherLevel(rotMapCollect[i*(sts.Nfilters-1)], params))
		// }
		if i == 0 {
			mErrCollect = rotMapCollect[0]
		} else {
			evaluator.Add(mErrCollect, rotMapCollect[i*(sts.Nfilters-1)], mErrCollect)
		}
	}

	// replicate the err k times and mult with weightsT
	rotIndsRep := utils.NegativeSlice(utils.NewSlice(0, (sts.Nfilters-1)*sts.Nclasses, sts.Nclasses))
	rotMapRep := evaluator.RotateHoisted(mErrCollect, rotIndsRep)

	mErrRep := rotMapRep[0]
	for i := 1; i < sts.Nfilters; i++ {
		evaluator.Add(mErrRep, rotMapRep[-i*sts.Nclasses], mErrRep)
	}

	// ------- min{u-5, inErr-3, b-3} level
	outErr := evaluator.MulRelinNew(mErrRep, weightsT)
	if err := evaluator.Rescale(outErr, params.Scale(), outErr); err != nil {
		panic("fail to rescale, dense backward outErr")
	}
	// fmt.Printf("min{u-5, inErr-3, b-3} " + utils.PrintCipherLevel(outErr, params))

	// 2. innerSum to get the err in nclasses * (0~k-1) slots, with garbage
	evaluator.InnerSum(outErr, 1, sts.Nclasses, outErr)

	return outErr
}

func (dense *Dense) Step(lr float64, momentum float64, eval ckks.Evaluator) bool {
	if dense.gradient != nil {
		update := eval.MultByConstNew(dense.gradient, lr)
		if dense.isMomentum {
			if dense.vt == nil {
				dense.vt = update.CopyNew().Ciphertext()
			} else {
				dense.vt = eval.MultByConstNew(dense.vt, momentum)
				dense.vt = eval.AddNew(dense.vt, update)
			}
			dense.weights = eval.SubNew(dense.weights, dense.vt)
		} else {
			dense.weights = eval.SubNew(dense.weights, update)
		}
		return true
	}
	return false
}

func (dense *Dense) PlainForwardCircuit(weights []complex128, input []complex128, sts *CellCnnSettings) ([]complex128, []complex128) {

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

func (dense *Dense) PlainBackwardCircuit(
	weights []complex128, input []complex128, u []complex128, err0 []complex128, sts *CellCnnSettings,
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
