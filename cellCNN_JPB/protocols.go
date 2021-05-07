package cellCNN

import(
	"github.com/ldsec/lattigo/v2/ckks"
	"math"
	"fmt"
)


type CellCNNProtocol struct{

	params *ckks.Parameters

	sk *ckks.SecretKey

	encoder ckks.Encoder 
	eval ckks.Evaluator

	mask []*ckks.Plaintext

	C, W *ckks.Matrix

	ctC, ctW *ckks.Ciphertext
	ctDC, ctDW *ckks.Ciphertext
	ctDCPrev, ctDWPrev *ckks.Ciphertext
	ctBoot *ckks.Ciphertext

	ptL []*ckks.Plaintext
	ptLTranspose []*ckks.Plaintext
}

func (c *CellCNNProtocol) Weights() (*ckks.Ciphertext, *ckks.Ciphertext){
	return c.ctC, c.ctW
}

func (c *CellCNNProtocol) UpdatedWeights() (*ckks.Ciphertext, *ckks.Ciphertext){
	return c.ctDC, c.ctDW
}

func (c *CellCNNProtocol) Eval() (ckks.Evaluator){
	return c.eval
}

func NewCellCNNProtocol(params *ckks.Parameters, sk *ckks.SecretKey) (c *CellCNNProtocol){
	c = new(CellCNNProtocol)

	c.params = params.Copy()

	c.sk = sk

	rlk, rotKey := GenPublicKeys(params, sk)

	c.encoder = ckks.NewEncoder(params)
	c.eval = ckks.NewEvaluator(params, ckks.EvaluationKey{rlk, rotKey})

	c.mask = GenPlaintextMaskForTrainingWithPrePooling(params, c.encoder)

	c.ptL = make([]*ckks.Plaintext, int(math.Ceil(float64(Features)*0.5)))
	for i := 0; i < int(math.Ceil(float64(Features)*0.5)); i++{
		c.ptL[i] = ckks.NewPlaintext(params, params.MaxLevel(), float64(params.Qi()[params.MaxLevel()-1]))
	}

	c.ptLTranspose = make([]*ckks.Plaintext, int(math.Ceil(float64(Samples)*0.5)))
	for i := 0; i < int(math.Ceil(float64(Samples)*0.5)); i++{
		c.ptLTranspose[i] = ckks.NewPlaintext(params, params.MaxLevel(), float64(params.Qi()[params.MaxLevel()-1]))
	}

	c.C = WeightsInit(Features, Filters, Features)
	c.W = WeightsInit(Filters, Classes, Filters) 

	c.ctC = EncryptRightForPtMul(c.C, BatchSize, 1, c.params, 4, sk)
	// [[ W transpose row encoded ] [         available         ]]
	//  |    classes * filters    | | Slots - classes * filters | 
	//
	c.ctW = EncryptRightForNaiveMul(c.W, BatchSize, c.params, 3, sk)

	return
}

func (c *CellCNNProtocol) Update(){

	c.ctDCPrev = c.ctDC.CopyNew().Ciphertext()
	c.ctDWPrev = c.ctDW.CopyNew().Ciphertext()

	c.eval.Sub(c.ctC, c.ctDC, c.ctC)
	c.eval.Sub(c.ctW, c.ctDW, c.ctW)
}

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

func (c *CellCNNProtocol) Backward(YBatch, XBatchTranspose *ckks.Matrix) {

	eval := c.eval
	params := c.params

	Y := EncodeLabelsForBackwardWithPrepooling(YBatch, Features, Filters, Classes, c.params)

	EncodeLeftForPtMul(XBatchTranspose, Filters, LearningRate, c.ptLTranspose, c.encoder, c.params) 

	convolutionMatrixSize := ConvolutionMatrixSize(BatchSize, Features, Filters)
	denseMatrixSize := DenseMatrixSize(Filters, Classes)

	// ctBoot
	//[[        U        ][           U           ] [      Ppool       ] [     W transpose row encoded     ] [ Previous DeltaW ] [        Previous DeltaC        ] [ available ]]
	// | DenseMatrixSize || classes*ConvolutionMatrixSize | | DenseMatrixSize  | | classes * ConvolutionMatrixSize | [ DenseMatrixSize ] [classes * ConvolutionMatrixSize] |           |

	// sigma(U) and sigma'(U)
	ctU1, ctU1Deriv := ActivationsCt(c.ctBoot, c.params, c.eval)

	// Y - sigma(U)
	eval.Sub(ctU1, Y, ctU1)

	// E1 = sigma'(U) * (sigma(U) - Y)
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
	eval.Mul(c.ctDC, c.mask[4], c.ctDC)

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

	// Adds DW with DWPrev*momentum 
	eval.Add(c.ctDC, c.ctDCPrev, c.ctDC)
	eval.Add(c.ctDW, c.ctDWPrev, c.ctDW)

	// Rescales
	eval.Rescale(c.ctDC, params.Scale(), c.ctDC)
	eval.Rescale(c.ctDW, params.Scale(), c.ctDW)
}

func (c *CellCNNProtocol) Refresh(){
	c.ctBoot = DummyBootWithPrepooling(c.ctBoot, c.params, c.sk)
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




func GenPublicKeys(params *ckks.Parameters, sk *ckks.SecretKey) (rlk *ckks.RelinearizationKey, rotKey *ckks.RotationKeySet){

	kgen := ckks.NewKeyGenerator(params)

	denseMatrixSize := DenseMatrixSize(Filters, Classes)
	convolutionMatrixSize := ConvolutionMatrixSize(BatchSize, Features, Filters)

	rotations := []int{}

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

	return kgen.GenRelinearizationKey(sk), kgen.GenRotationKeysForRotations(rotations, true, sk)
}