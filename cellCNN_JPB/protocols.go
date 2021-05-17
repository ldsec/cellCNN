package cellCNN

import(
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/drlwe"
	"github.com/ldsec/lattigo/v2/rlwe"
	"math"
	"fmt"
)

// CellCNNProtocol is a structure storing all the necessary elements and ciphertext
// for the cellCNN protocol
type CellCNNProtocol struct{


	params *ckks.Parameters

	sk *ckks.SecretKey
	pk *ckks.PublicKey
	rlk *ckks.RelinearizationKey
	rotKey *ckks.RotationKeySet

	rotKeyIndex []int

	encryptor ckks.Encryptor
	decryptor ckks.Decryptor
	encoder ckks.Encoder 
	eval ckks.Evaluator

	mask []*ckks.Plaintext

	C, W *ckks.Matrix
	DC, DW *ckks.Matrix
	DCPrev, DWPrev *ckks.Matrix
	P, U *ckks.Matrix

	ctC, ctW *ckks.Ciphertext
	ctDC, ctDW *ckks.Ciphertext
	ctDCPrev, ctDWPrev *ckks.Ciphertext
	ctBoot *ckks.Ciphertext

	ptL []*ckks.Plaintext
	ptLTranspose []*ckks.Plaintext

	ckgProtocol *dckks.CKGProtocol
	rkgProtocol *dckks.RKGProtocol
	rtgProtocol *dckks.RTGProtocol
	cksProtocol *dckks.CKSProtocol

	ckgShare    *drlwe.CKGShare
	rkgEphemSk  *rlwe.SecretKey
	rkgShareOne *drlwe.RKGShare
	rkgShareTwo *drlwe.RKGShare
	rtgShare    *drlwe.RTGShare
	cksShare    dckks.CKSShare
}


func (c *CellCNNProtocol) PK() (*ckks.PublicKey){
	return c.pk
}

func (c *CellCNNProtocol) SK() (*ckks.SecretKey){
	return c.sk
}

func (c *CellCNNProtocol) RotKeyIndex() []int{
	return c.rotKeyIndex
}

func (c *CellCNNProtocol) HasRotKey(galEl uint64) bool{
	_, ok := c.rotKey.Keys[galEl]
	return ok
}

// W returns the dense layer matrix
func (c *CellCNNProtocol) CtW() (*ckks.Ciphertext){
	return c.ctW
}

func (c *CellCNNProtocol) EvaluatorInit(){
	c.eval = ckks.NewEvaluator(c.params, ckks.EvaluationKey{c.rlk, c.rotKey})
}

// C returns the convolution matrix
func (c *CellCNNProtocol) CtC() (*ckks.Ciphertext){
	return c.ctC
}

func (c *CellCNNProtocol) CtBoot() (*ckks.Ciphertext){
	return c.ctBoot
}

// SecretKey returns the secret-key
func (c *CellCNNProtocol) SecretKey() (*ckks.SecretKey){
	return c.sk
}

// DW returns the updated weights of the dense layer matrix
func (c *CellCNNProtocol) CtDW() (*ckks.Ciphertext){
	return c.ctDW
}

// DC returns the updated weights of the convolution matrix
func (c *CellCNNProtocol) CtDC() (*ckks.Ciphertext){
	return c.ctDC
}

func (c *CellCNNProtocol) Encoder() (ckks.Encoder){
	return c.encoder
}

// Eval returns the evaluator
func (c *CellCNNProtocol) Eval() (ckks.Evaluator){
	return c.eval
}

func (c *CellCNNProtocol) PrintCtWPrecision(){

		fmt.Println("W")
		//c.W.Transpose().Print()
		w := []complex128{}
		for i := 0; i < Classes; i++{
			y := c.encoder.Decode(c.decryptor.DecryptNew(c.eval.RotateNew(c.ctW, i*BatchSize*Filters)), c.params.LogSlots())
			w = append(w, y[:Filters]...)
		}
		W := ckks.NewMatrix(Classes, Filters)
		W.M = w

		//W.Print()

		precisionStats := ckks.GetPrecisionStats(c.params, c.encoder, nil, c.W.Transpose().M, W.M, 0)
		fmt.Println(precisionStats.String())

}

func (c *CellCNNProtocol) PrintCtCPrecision(){
	fmt.Println("C")
	//c.C.Print()
	y := c.encoder.Decode(c.decryptor.DecryptNew(c.ctC), c.params.LogSlots())

	C := ckks.NewMatrix(Features, Filters)
	C.M = y[:Filters*Features]

	//C.Print()

	precisionStats := ckks.GetPrecisionStats(c.params, c.encoder, nil, c.C.M, C.M, 0)
	fmt.Println(precisionStats.String())
}

// NewCellCNNProtocol creates a new cellCNN protocol
func NewCellCNNProtocol(params *ckks.Parameters) (c *CellCNNProtocol){
	c = new(CellCNNProtocol)

	c.params = params

	kgen := ckks.NewKeyGenerator(params)

	c.sk = kgen.GenSecretKey()

	c.rotKeyIndex = GetRotationKeysIndex(kgen)

	
	c.decryptor = ckks.NewDecryptor(params, c.sk)
	c.encoder = ckks.NewEncoder(params)
	
	c.mask = c.genPlaintextMaskForTrainingWithPrePooling(params)

	c.ptL = make([]*ckks.Plaintext, int(math.Ceil(float64(Features)*0.5)))
	for i := 0; i < int(math.Ceil(float64(Features)*0.5)); i++{
		c.ptL[i] = ckks.NewPlaintext(params, params.MaxLevel(), float64(params.Qi()[params.MaxLevel()-1]))
	}

	c.ptLTranspose = make([]*ckks.Plaintext, int(math.Ceil(float64(Samples)*0.5)))
	for i := 0; i < int(math.Ceil(float64(Samples)*0.5)); i++{
		c.ptLTranspose[i] = ckks.NewPlaintext(params, params.MaxLevel(), float64(params.Qi()[params.MaxLevel()-1]))
	}

	c.DW = new(ckks.Matrix)
	c.DC = new(ckks.Matrix)
	c.P = new(ckks.Matrix)
	c.U = new(ckks.Matrix)
	return
}


func (c *CellCNNProtocol) SetWeights(C, W *ckks.Matrix){
	c.C = C.Copy()
	c.W = W.Copy()
}

func (c *CellCNNProtocol) EncryptWeights(){

	c.encryptor = ckks.NewEncryptorFromPk(c.params, c.pk)

	c.ctC = EncryptRightForPtMul(c.C, BatchSize, 1, c.params, 4, c.encoder, c.encryptor)
	// [[ W transpose row encoded ] [         available         ]]
	//  |    classes * filters    | | Slots - classes * filters | 
	//
	c.ctW = EncryptRightForNaiveMul(c.W, BatchSize, c.params, 3, c.encoder, c.encryptor)

	c.encryptor = nil
}

func (c *CellCNNProtocol) genPlaintextMaskForTrainingWithPrePooling(params *ckks.Parameters) []*ckks.Plaintext{

	denseMatrixSize := DenseMatrixSize(Filters, Classes)
	convolutionMatrixSize := ConvolutionMatrixSize(BatchSize, Features, Filters)

	levelMaskPtW := 5
	levelMaskPtC := 5

	scaleMaskPtW := float64(params.Qi()[levelMaskPtW])
	scaleMaskPtC := float64(params.Qi()[levelMaskPtC])

	// Mask W
	maskW := make([]complex128, params.Slots())

	// mask
	for i := 0; i < BatchSize*denseMatrixSize; i++ {
		maskW[i] = complex(1.0, 0)
	}
	maskPtW := ckks.NewPlaintext(params, levelMaskPtW, scaleMaskPtW)
	c.encoder.EncodeNTT(maskPtW, maskW, params.LogSlots())

	// mask avg w0
	maskW = make([]complex128, params.Slots())
	for i := 0; i < denseMatrixSize>>1; i++ {
		maskW[i] = complex(1.0, 0)
		maskW[i+(denseMatrixSize>>1)] = complex(0, 0)
	}
	maskPtW0 := ckks.NewPlaintext(params, levelMaskPtW, scaleMaskPtW)
	c.encoder.EncodeNTT(maskPtW0, maskW, params.LogSlots())

	// mask avg w1
	maskW = make([]complex128, params.Slots())
	for i := 0; i < denseMatrixSize>>1; i++ {
		maskW[i] = complex(0, 0)
		maskW[(denseMatrixSize>>1)+i] = complex(1.0, 0)
	}

	maskPtW1 := ckks.NewPlaintext(params, levelMaskPtW, scaleMaskPtW)
	c.encoder.EncodeNTT(maskPtW1, maskW, params.LogSlots())

	// Mask C
	maskC := make([]complex128, params.Slots())
	for i := 0; i < convolutionMatrixSize; i++ {
		maskC[i] = complex(1.0, 0)
	}
	maskPtC := ckks.NewPlaintext(params, levelMaskPtC, scaleMaskPtC)
	c.encoder.EncodeNTT(maskPtC, maskC, params.LogSlots())

	// mask with avg and 0.5 factor for the imaginary part removal
	maskC = make([]complex128, params.Slots())
	for i := 0; i < convolutionMatrixSize; i++ {
		maskC[i] = complex(0.5, 0)
	}
	maskPtCHalf := ckks.NewPlaintext(params, levelMaskPtC, scaleMaskPtC)
	c.encoder.EncodeNTT(maskPtCHalf, maskC, params.LogSlots())

	return []*ckks.Plaintext{maskPtW, maskPtW0, maskPtW1, maskPtC, maskPtCHalf}
}



func (c *CellCNNProtocol) ForwardPlain(XBatch *ckks.Matrix){
	// Convolution
	c.P.MulMat(XBatch, c.C)

	// Dense
	c.U.MulMat(c.P, c.W)

}

func (c *CellCNNProtocol) BackWardPlain(XBatch, YBatch *ckks.Matrix){

	L1Batch := new(ckks.Matrix)
	L1DerivBatch := new(ckks.Matrix)
	E0Batch := new(ckks.Matrix)
	E1Batch := new(ckks.Matrix)

	// Activations
	L1Batch.Func(c.U, Activation)
	L1DerivBatch.Func(c.U, ActivationDeriv)

	// Dense error
	E1Batch.Sub(L1Batch, YBatch)
	E1Batch.Dot(E1Batch, L1DerivBatch)

	// Convolution error
	E0Batch.MulMat(E1Batch, c.W.Transpose())

	// Updated weights
	c.DW.MulMat(c.P.Transpose(), E1Batch)
	c.DC.MulMat(XBatch.Transpose(), E0Batch)

	c.DW.MultConst(c.DW, complex(LearningRate, 0))
	c.DC.MultConst(c.DC, complex(LearningRate, 0))

	if c.DCPrev == nil{
		c.DCPrev = ckks.NewMatrix(Features, Filters)
	}

	if c.DWPrev == nil{
		c.DWPrev = ckks.NewMatrix(Filters, Classes)
	}

	// Adds the previous weights
	// W_i = learning_rate * Wt + W_i-1 * momentum
	c.DW.Add(c.DW, c.DWPrev)
	c.DC.Add(c.DC, c.DCPrev)

	// Stores the current weights
	// W_i = learning_rate * Wt + W_i-1 * momentum
	c.DWPrev.MultConst(c.DW, complex(Momentum, 0))
	c.DCPrev.MultConst(c.DC, complex(Momentum, 0))
}

func (c *CellCNNProtocol) UpdatePlain(DC, DW *ckks.Matrix){
	// Updates the matrices
	c.W.Sub(c.W, DW)
	c.C.Sub(c.C, DC)
}

// Forward applies a forward pass on the given batch of samples and stores the result in ctBoot
func (c *CellCNNProtocol) Forward(XBatch *ckks.Matrix){

	// Encodes the Batch
	EncodeLeftForPtMul(XBatch, Filters, 1.0, c.ptL, c.encoder, c.params)

	// Convolution
	ctPpool := Convolution(c.ptL, c.ctC, Features, Filters, c.eval)

	// Replicates the values for all the classes
	c.eval.Replicate(ctPpool, BatchSize*Filters, Classes, ctPpool)

	// Dense Layer
	c.ctBoot = DenseLayer(ctPpool, c.ctW, Filters, Classes, c.eval)
	
	// Repacking for bootstrapping
	RepackBeforeBootstrappingWithPrepooling(c.ctBoot, ctPpool, c.ctW, c.ctDCPrev, c.ctDWPrev, BatchSize, Filters, Classes, c.eval)
}

func (c *CellCNNProtocol) PredictPlain(XBatch *ckks.Matrix) (*ckks.Matrix){

	// Convolution
	c.P.MulMat(XBatch, c.C)

	U := new(ckks.Matrix)

	// Dense
	U.MulMat(c.P, c.W)

	// Activations
	U.Func(U, Activation)

	return U
}


func (c *CellCNNProtocol) Predict(XBatch *ckks.Matrix) (*ckks.Matrix){

	// Encodes the Batch
	EncodeLeftForPtMul(XBatch, Filters, 1.0, c.ptL, c.encoder, c.params)

	// Convolution
	ctP := Convolution(c.ptL, c.ctC, Features, Filters, c.eval)

	// Replicates the values for all the classes
	c.eval.Replicate(ctP, BatchSize*Filters, Classes, ctP)

	// Dense Layer
	ctU := DenseLayer(ctP, c.ctW, Filters, Classes, c.eval)

	// pow2 * 0.5 (for the complex trick)
	pow2 := math.Exp2(math.Round(math.Log2(float64(c.params.Qi()[0]))))/2.0

	c.eval.MultByConst(ctU, int(pow2/ctU.Scale()), ctU)
	ctU.SetScale(pow2*2)

	c.eval.Add(ctU, c.eval.ConjugateNew(ctU), ctU)

	var ctPredict *ckks.Ciphertext
	var err error
	if ctPredict, err = c.eval.EvaluatePoly(ctU, ckks.NewPoly(coeffsActivation), c.params.Scale()); err != nil {
		panic(err)
	}

	res := c.encoder.Decode(c.decryptor.DecryptNew(ctPredict), c.params.LogSlots())

	U := ckks.NewMatrix(BatchSize, Classes)

	for i := 0; i < BatchSize; i++{
		for j := 0; j < Classes; j++{
			U.M[i*Classes+j] = res[i*Filters+BatchSize*Filters*j]
		}
	}

	return U
}

// Backward applies a backward pass on the given batch and stores the result in ctDC and ctDW
func (c *CellCNNProtocol) Backward(XBatch, YBatch *ckks.Matrix) {

	eval := c.eval
	params := c.params

	Y := EncodeLabelsForBackwardWithPrepooling(YBatch, Features, Filters, Classes, c.params)

	EncodeLeftForPtMul(XBatch.Transpose(), Filters, LearningRate*0.5, c.ptLTranspose, c.encoder, c.params) 

	convolutionMatrixSize := ConvolutionMatrixSize(BatchSize, Features, Filters)
	denseMatrixSize := DenseMatrixSize(Filters, Classes)

	// ctBoot
	//[[        U        ][           U           ] [      Ppool       ] [     W transpose row encoded     ] [ Previous DeltaW ] [        Previous DeltaC        ] [ available ]]
	// | DenseMatrixSize || classes*ConvolutionMatrixSize | | DenseMatrixSize  | | classes * ConvolutionMatrixSize | [ DenseMatrixSize ] [classes * ConvolutionMatrixSize] |           |

	// Lvl 7 sigma(U) and sigma'(U)
	ctU1, ctU1Deriv := ActivationsCt(c.ctBoot, c.params, c.eval)

	// Y - sigma(U)
	eval.Sub(ctU1, Y, ctU1)

	// Lvl 6 E1 = sigma'(U) * (sigma(U) - Y)
	eval.MulRelin(ctU1, ctU1Deriv, ctU1)
	if err := eval.Rescale(ctU1, params.Scale(), ctU1); err != nil{
		panic(err)
	}

	// Access the index of the pooling results and upsampled W

	//[[      Ppool       ] [     W transpose row encoded    ] [ Previous DeltaW ] [     Previous DeltaC           ] [ available ] [        U        ][                U                ] ]
	// | DenseMatrixSize  | | classes * ConvolutionMatrixSize| [ DenseMatrixSize ] [classes * ConvolutionMatrixSize] |           | | DenseMatrixSize || classes * ConvolutionMatrixSize | 
	c.ctDW = eval.RotateNew(c.ctBoot, BatchSize*denseMatrixSize + Classes*convolutionMatrixSize)

	// Multiplies at the same time pool^t x E1 and upSampled(E1 x W^t)
	eval.MulRelin(c.ctDW, ctU1, c.ctDW)
	if err := eval.Rescale(c.ctDW, params.Scale(), c.ctDW); err != nil{
		panic(err)
	}

	// Accesses upSampled(E1 x W^t)
	c.ctDC = eval.RotateNew(c.ctDW, BatchSize*denseMatrixSize)

	// Sums accroses the batches
	// [        W0       ] [       W1        ] [            Garbage              ]  
	// | denseMatrixSize | | denseMatrixSize | | 2*(batches-1) * denseMatrixSize | 
	eval.InnerSum(c.ctDW, Classes*Filters, BatchSize, c.ctDW)

	// Sums accrosses the classes
	eval.InnerSum(c.ctDC, convolutionMatrixSize, Classes, c.ctDC)
	
	//  DC = Ltranspose x E0
	c.ctDC = MulMatrixLeftPtWithRightCt(c.ptLTranspose, c.ctDC, BatchSize, Filters, c.eval)

	// Extracts previous DC * momentum and previous DW * momentum

	//[[       Previous DeltaC            ] [ available ] [        U        ][                U                ] [      Ppool       ] [     W transpose row encoded    ] [ Previous DeltaW ] ]
	// [ classes * ConvolutionMatrixSize  ] |           | | DenseMatrixSize || classes * ConvolutionMatrixSize | | DenseMatrixSize  | | classes * ConvolutionMatrixSize| [ DenseMatrixSize ] 
	c.ctDCPrev = eval.RotateNew(c.ctBoot, 3*BatchSize*denseMatrixSize + 2*Classes*convolutionMatrixSize)

	//[[ Previous DeltaW ] [     Previous DeltaC           ] [ available ] [        U        ][                U                ] [      Ppool       ] [     W transpose row encoded    ] ]
	// [ DenseMatrixSize ] [classes * ConvolutionMatrixSize] |           | | DenseMatrixSize || classes * ConvolutionMatrixSize | | DenseMatrixSize  | | classes * ConvolutionMatrixSize| 
	c.ctDWPrev = eval.RotateNew(c.ctBoot, 2*BatchSize*denseMatrixSize + 2*Classes*convolutionMatrixSize)

	// Cleans the imaginary part
	eval.Add(c.ctDC, eval.ConjugateNew(c.ctDC), c.ctDC)

	// Replicates ctDC so that it is at least as large as convolutionMatrixSize
	eval.Replicate(c.ctDC, Features*Filters, int(math.Ceil(float64(convolutionMatrixSize)/float64(Features * Filters))), c.ctDC)

	// Divides by the average and learning rate and cleans the non-desired slots

	// Divides by the average, masks the values and extract the first and second classe
	ctDWtmp := eval.MulNew(c.ctDW, c.mask[1])
	eval.Mul(c.ctDW, c.mask[2], c.ctDW)

	// Replicates DW batch times (no masking needed as it is a multiple of filters)
	eval.Rotate(c.ctDW, -BatchSize*Filters + Filters, c.ctDW)
	eval.Add(c.ctDW, ctDWtmp, c.ctDW)
	eval.Replicate(c.ctDW, Filters, BatchSize, c.ctDW)

	// Mask DWPrev*momentum and DCPrev*momentum
	eval.Mul(c.ctDCPrev, c.mask[3], c.ctDCPrev)
	eval.Mul(c.ctDWPrev, c.mask[0], c.ctDWPrev)

	eval.Rescale(c.ctDCPrev, params.Scale(), c.ctDCPrev)

	// Adds DW with DWPrev*momentum 
	eval.Add(c.ctDC, c.ctDCPrev, c.ctDC)
	eval.Add(c.ctDW, c.ctDWPrev, c.ctDW)

	// Rescales
	eval.Rescale(c.ctDW, params.Scale(), c.ctDW)
}

// Refresh refreshes and repack ctBoot using DummyBootWithPrepooling
func (c *CellCNNProtocol) Refresh(){
	c.ctBoot = DummyBootWithPrepooling(c.ctBoot, c.params, c.sk)
}

// Update stores the input DC and DW, and updated the weights of the convolution and dense matrix
func (c *CellCNNProtocol) Update(DC, DW *ckks.Ciphertext){

	c.ctDCPrev = DC.CopyNew().Ciphertext()
	c.ctDWPrev = DW.CopyNew().Ciphertext()

	c.eval.Sub(c.ctC, DC, c.ctC)
	c.eval.Sub(c.ctW, DW, c.ctW)
}

func DummyBootWithPrepooling(ciphertext *ckks.Ciphertext, params *ckks.Parameters, sk *ckks.SecretKey) (*ckks.Ciphertext){

	//  [            CTU            ] [      CTPpool       ] [             CTW            ] [         prevCTDW           ] [                         prevCTDC                        ] [available] [   garbage  ]
	//  | batches * DenseMatrixSize | | batches * Features | | batches *  DenseMatrixSize | | batches *  DenseMatrixSize | | batches * Filters + (Features/2 -1)*2*Filters + Filters | |         | | Filters -1 |
	//

	//DecryptPrint(2*BatchSize, Filters, true, ciphertext, params, sk)

	convolutionMatrixSize := ConvolutionMatrixSize(BatchSize, Features, Filters)
	denseMatrixSize := DenseMatrixSize(Filters, Classes)

	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptorFromSk(params, sk)

	v := encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())

	newv := make([]complex128, params.Slots())

	//[[            U            ] [ available ]]
	// | batches*Classes*Filters | |           | 

	// Reorder the slots (groupes samples together) from
	//
	// [ S0U0 ] ... [ SiU0 ] [ S0U1 ] ... [ SiU1 ] ]
	//
	// to
	//
	// [ S0U0 ] [ S0U1 ] ... [ SiU0 ] ... [ SiU1 ] ]
	// 
	for i := 0; i < BatchSize; i++ {
		for j := 0; j < Classes; j++{
			c := complex(real(v[i*Filters+j*Filters*BatchSize]), 0)
			for k := 0; k < Filters; k++{
				newv[i*Filters*Classes + j*Filters + k] = c
			}
		}	
	}

	idx := BatchSize*denseMatrixSize

	// Reorders slots (groupes samples tougether)
	//[[            U            ] [                U                ] [ available ]]
	// | batches*Classes*Filters | | Classes * convolutionMatrixSize | |           | 
	//
	// From 
	//
	// [ S0U0 ] ... [ SiU0 ] [ S0U1 ] ... [ SiU1 ] ]
	//
	// to
	//
	// [ S0U0 ] [ S0U1 ] ... [ SiU0 ] ... [ SiU1 ] ]
	// 
	// Aranges slots such that U for classe 0 of each sample is replicated Filters times
	// cycling through the samples, but up to convolutionMatrixSize times
	// Repeates the process for U for classe 1.
	for i := 0; i < Classes; i++ {
		pos := 0
		for j := 0; j < convolutionMatrixSize; j++ {
			c := complex(real(v[i*BatchSize*Filters + ((pos*Filters)%(BatchSize*Filters))]), 0)

			if (j+1)%Filters == 0{
				pos++
			}

			newv[idx + i*convolutionMatrixSize + j] = c
		}
	}

	idx += Classes * convolutionMatrixSize


	//[[            U            ] [                U                ] [         Ppool           ] [ available ]]
	// | batches*Classes*Filters | | Classes * convolutionMatrixSize | | batches*Classes*Filters | |           |

	// Reorder the slots (groupes samples together) from
	//
	// [ S0P0 ] ... [ SiP0 ] ... [ S0P0 ] ... [ SiP0 ]
	// to
	// [ S0P0 ] ... [ S0P0 ] ... [ SiP0 ] ... [ SiP0 ]
	// | Classes * Filters | ... | Classes * Filters |
	for i := 0; i < BatchSize; i++{
		for j := 0; j < Filters; j++ {
			c := complex(real(v[BatchSize*denseMatrixSize + i*Filters + j]) * LearningRate, 0)
			for k := 0; k < Classes; k++ {
				newv[idx + i*Filters*Classes + k*Filters + j] = c
			}
		}
	}
	
	idx += BatchSize * denseMatrixSize

	//[[            U            ] [                U                ] [         Ppool           ] [    W transpose row encoded      ] [ available ]]
	// | batches*Classes*Filters | | Classes * convolutionMatrixSize | | batches*Classes*Filters | | Classes * convolutionMatrixSize | |           |
	for i := 0; i < Classes; i++ {

		for j := 0; j < convolutionMatrixSize; j++ {
			c := real(v[2*BatchSize*denseMatrixSize + i * Filters*BatchSize + (j%Filters)])
			newv[idx + convolutionMatrixSize*i + j] = complex(c, 0)
		}
	}

	idx += Classes * convolutionMatrixSize

	// Copies the previous weights and multiples them by the momemtum
	//[[             U           ] [                U                ] [          Ppool           ] [      W transpose row encoded    ] [     Previous DeltaW     ] [   Previous DeltaC     ] [ available ]]
	// | batches*Classes*Filters | | Classes * convolutionMatrixSize | | batches*Classes*Filters  | | Classes * convolutionMatrixSize | | batches*Classes*Filters | | convolutionMatrixSize |
	for i := 0; i < BatchSize * denseMatrixSize; i++ {
		newv[idx + i] = complex(real(v[3*BatchSize*denseMatrixSize+i])*Momentum, 0)
	}

	idx += BatchSize * denseMatrixSize

	for i := 0; i < convolutionMatrixSize; i++ {
		newv[idx + i] = complex(real(v[4*BatchSize*denseMatrixSize+i])*Momentum, 0)
	}

	idx += convolutionMatrixSize

	if false {
		fmt.Println("Repacked Plaintext")
		for i := 0; i < idx; i++{
			fmt.Println(i, newv[i])
		}
	}

	pt := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	encoder.EncodeNTT(pt, newv, params.LogSlots())
	newCt := encryptor.EncryptNew(pt)

	return newCt

}




func GetRotationKeysIndex(kgen ckks.KeyGenerator) (rotations []int){

	denseMatrixSize := DenseMatrixSize(Filters, Classes)
	convolutionMatrixSize := ConvolutionMatrixSize(BatchSize, Features, Filters)

	rotations = []int{}

	rotations = append(rotations, Filters)

	// Convolution rotations
	for i := 1; i < Features>>1; i++ {
		rotations = append(rotations, 2*Filters*i)
	}

	for i := 1; i < BatchSize>>1; i++ {
		rotations = append(rotations, 2*Filters*i)
	}

	// Dense layer rotations
	rotations = append(rotations, kgen.GenRotationIndexesForInnerSum(1, Filters)...)

	rotations = append(rotations, kgen.GenRotationIndexesForInnerSum(convolutionMatrixSize, Classes)...)

	rotations = append(rotations, kgen.GenRotationIndexesForInnerSum(BatchSize, Classes)...)

	rotations = append(rotations, kgen.GenRotationIndexesForInnerSum(Classes*Filters, BatchSize)...)

	// Pre-pool convolution replication
	rotations = append(rotations, -BatchSize*Filters)

	// Repacking of ctPpool before bootstrapping
	rotations = append(rotations,  1*BatchSize*denseMatrixSize)
	rotations = append(rotations, -1*BatchSize*denseMatrixSize)
	rotations = append(rotations, -2*BatchSize*denseMatrixSize)
	rotations = append(rotations, -3*BatchSize*denseMatrixSize)
	rotations = append(rotations, -4*BatchSize*denseMatrixSize)

	rotations = append(rotations, 1*BatchSize*denseMatrixSize + 1*Classes*convolutionMatrixSize)
	rotations = append(rotations, 1*BatchSize*denseMatrixSize + 2*Classes*convolutionMatrixSize)
	rotations = append(rotations, 2*BatchSize*denseMatrixSize + 2*Classes*convolutionMatrixSize)
	rotations = append(rotations, 3*BatchSize*denseMatrixSize + 2*Classes*convolutionMatrixSize)

	rotations = append(rotations, -BatchSize*Filters + Filters)

	// Replication of DC
	rotations = append(rotations, kgen.GenRotationIndexesForReplicate(Features*Filters, int(math.Ceil(float64(convolutionMatrixSize)/float64(Features * Filters))))...)

	// Replication of DW
	rotations = append(rotations, kgen.GenRotationIndexesForReplicate(Filters, BatchSize)...)

	for i := 0; i < Classes; i++{
		rotations = append(rotations, i*BatchSize*Filters)
	}

	return rotations
}