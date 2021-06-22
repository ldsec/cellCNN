package centralized

import (
	"time"

	"github.com/ldsec/cellCNN/cellcnnPoseidon/layers"
	"github.com/ldsec/cellCNN/cellcnnPoseidon/utils"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"gonum.org/v1/gonum/mat"
)

type CellCNN struct {
	// network settings
	cnnSettings *utils.CellCnnSettings
	// crypto settings
	params    ckks.Parameters
	relikey   *rlwe.RelinearizationKey
	encoder   ckks.Encoder
	encryptor ckks.Encryptor
	// layers
	conv1d    *layers.Conv1D
	dense     *layers.Dense
	evaluator ckks.Evaluator
	btp       *ckks.Bootstrapper
	sk        *rlwe.SecretKey
	pcir      *PlainCircuit
	momentum  float64
	lr        float64
}

func (c *CellCNN) GetEncoder() ckks.Encoder {
	return c.encoder
}

func (c *CellCNN) GetEvaluator() ckks.Evaluator {
	return c.evaluator
}

func (c *CellCNN) GetWeights() []*ckks.Ciphertext {
	econv := c.conv1d.GetWeights()
	edense := c.dense.GetWeights()
	return append(econv, edense)
}

func (c *CellCNN) GetGradients() *Gradients {
	return &Gradients{c.conv1d.GetGradient(), c.dense.GetGradient()}

}

type PlainCircuit struct {
	// weights
	filters [][]complex128
	weights []complex128

	// copmutation results
	input    []complex128
	actv     []complex128
	pred     []complex128
	dfilters [][]complex128
	dweights []complex128
	u        []complex128
}

func NewPlainCircuit(filters [][]complex128, weights []complex128, input []complex128) *PlainCircuit {
	return &PlainCircuit{
		filters: filters,
		weights: weights,
		input:   input,
	}
}

func NewCellCNN(sts *utils.CellCnnSettings, cryptoParams *utils.CryptoParams, momentum, lr float64) *CellCNN {

	model := &CellCNN{
		cnnSettings: sts,
		params:      cryptoParams.Params,
		relikey:     cryptoParams.Rlk,
		encoder:     cryptoParams.GetEncoder(),
		encryptor:   cryptoParams.GetEncryptor(),
		momentum:    momentum,
		lr:          lr,
	}

	return model
}

func (c *CellCNN) UpdateWithGradients(g *Gradients) {
	c.conv1d.UpdateWithGradients(g.filters, c.evaluator)
	c.dense.UpdateWithGradients(g.dense, c.evaluator)
}

func (c *CellCNN) Marshall() (data [][]byte) {
	filterData := c.conv1d.Marshall()
	denseData := c.dense.Marshall()
	// milestone = len(filterData)
	data = append(filterData, denseData)
	return
}

func (c *CellCNN) Unmarshall(data [][]byte) {
	nfilters := len(data) - 1
	c.conv1d.Unmarshall(data[:nfilters])
	c.dense.Unmarshall(data[nfilters])
}

// return the the graident
func (c *CellCNN) GetGradient() []*ckks.Ciphertext {
	filters := c.conv1d.GetGradient()
	dense := c.dense.GetGradient()
	return append(filters, dense)
}

// return the byte representation of the graident
func (c *CellCNN) GetGradientBinary() [][]byte {
	dConv := c.conv1d.GetGradientBinary()
	dDense := c.dense.GetGradientBinary()
	return append(dConv, dDense)
}

func (c *CellCNN) WithEvaluator(eval ckks.Evaluator) {
	c.evaluator = eval
}

func (c *CellCNN) WithSk(sk *rlwe.SecretKey) {
	c.sk = sk
}

func (c *CellCNN) WithDiagM(diagM *ckks.PtDiagMatrix) {
	c.dense.WithDiagM(diagM)
}

func (c *CellCNN) FisrtMomentum() bool {
	return c.conv1d.FirstMomentum() && c.dense.FirstMomentum()
}

func (c *CellCNN) UpdateMomentum(grad *Gradients) {
	c.conv1d.UpdateMomentum(grad.filters)
	c.dense.UpdateMomentum(grad.dense)
}

// InitWeights init CellCNN and return the init plaintext weights (conv, dense) as mat.Dense
func (c *CellCNN) InitWeights(
	wConv1D []*ckks.Ciphertext, wDense *ckks.Ciphertext,
	pConv, pDense []complex128,
) (*mat.Dense, *mat.Dense) {
	nfilters := c.cnnSettings.Nfilters
	nmakers := c.cnnSettings.Nmakers
	nclasses := c.cnnSettings.Nclasses
	ncells := c.cnnSettings.Ncells
	encoder := c.encoder
	encryptor := c.encryptor

	// generate random conv1d weights
	if wConv1D == nil {
		wConv1D = make([]*ckks.Ciphertext, 0)
		// var plain []complex128
		for i := 0; i < nfilters; i++ {
			plain, tmp := utils.WeightsInit(int(c.params.Slots()), nmakers, float64(nmakers), ncells)
			pConv = append(pConv, tmp...)
			encoded := encoder.EncodeNTTAtLvlNew(c.params.MaxLevel(), plain, c.params.LogSlots())
			encrpted := encryptor.EncryptNew(encoded)
			wConv1D = append(wConv1D, encrpted)
		}
	}

	// generate random dense weights
	if wDense == nil {
		var plainDense []complex128
		plainDense, pDense = utils.WeightsInit(int(c.params.Slots()), nfilters*nclasses, float64(nfilters), 1)
		encodeDense := encoder.EncodeNTTAtLvlNew(c.params.MaxLevel(), plainDense, c.params.LogSlots())
		wDense = encryptor.EncryptNew(encodeDense)
	}

	var cm, dm *mat.Dense

	if pConv != nil {
		cw := make([]float64, len(pConv))
		for i := 0; i < len(pConv); i++ {
			cw[i] = real(pConv[i])
		}
		cwNew := make([]float64, len(pConv))
		for i := 0; i < nmakers; i++ {
			for j := 0; j < nfilters; j++ {
				cwNew[i*nfilters+j] = cw[j*nmakers+i]
			}
		}
		cm = mat.NewDense(nmakers, nfilters, cwNew)
	}

	if pDense != nil {
		dw := make([]float64, len(pDense))
		for i := 0; i < len(pDense); i++ {
			dw[i] = real(pDense[i])
		}
		dwNew := make([]float64, len(pDense))
		for i := 0; i < nfilters; i++ {
			for j := 0; j < nclasses; j++ {
				dwNew[i*nclasses+j] = dw[j*nfilters+i]
			}
		}
		dm = mat.NewDense(nfilters, nclasses, dwNew)
	}

	c.conv1d = layers.NewConv1D(wConv1D, c.momentum)
	c.conv1d.WithEncoder(encoder)
	c.dense = layers.NewDense(wDense, c.momentum)
	c.dense.WithEncoder(encoder)
	return cm, dm
}

func (c *CellCNN) InitEvaluator(cryptoParams *utils.CryptoParams, maxM1N2Ratio float64) ckks.Evaluator {
	kgen := cryptoParams.Kgen()
	encoder := c.encoder
	params := cryptoParams.Params
	sk := cryptoParams.Sk

	Cinds := c.conv1d.InitRotationInds(c.cnnSettings, params)
	Dinds := c.dense.InitRotationInds(c.cnnSettings, kgen, c.params, encoder, maxM1N2Ratio)
	Rinds := utils.ClearRotInds(append(Cinds, Dinds...), params.Slots())

	rks := kgen.GenRotationKeysForRotations(Rinds, false, sk)
	c.evaluator = ckks.NewEvaluator(c.params, rlwe.EvaluationKey{Rlk: c.relikey, Rtks: rks})
	return c.evaluator
}

func (c *CellCNN) GenerateMaskMap() map[int]*ckks.Plaintext {
	nfilters := c.cnnSettings.Nfilters
	nclasses := c.cnnSettings.Nclasses
	// dense maskMap to collect all results into one ciphertext
	maskMap := make(map[int]*ckks.Plaintext)
	for i := 0; i < nclasses; i++ {
		maskMap[i*(nfilters-1)] = func() *ckks.Plaintext {
			tmpMask := make([]complex128, c.params.Slots())
			// fmt.Println("making maskMap: ", i*(nfilters-1))
			tmpMask[i] = complex(float64(1), 0)
			return c.encoder.EncodeNTTAtLvlNew(c.params.MaxLevel(), tmpMask, c.params.LogSlots())
		}()
	}
	return maskMap
}

// ForwardOne return the ciphertext predition and the time consumed on layer1, layer2, sum of two
func (c *CellCNN) ForwardOne(
	input *ckks.Plaintext,
	wConv []*ckks.Ciphertext,
	wDense *ckks.Ciphertext,
	poolMask *ckks.Plaintext,
	// maskMap map[int]*ckks.Plaintext,
) (*ckks.Ciphertext, []float64) {
	t1 := time.Now()
	out1 := c.conv1d.Forward(input, wConv, c.cnnSettings, c.evaluator, c.params)
	t2 := time.Now()
	out2 := c.dense.Forward(out1, wDense, c.cnnSettings, c.evaluator, c.encoder, c.params)
	t3 := time.Now()
	return out2, []float64{t2.Sub(t1).Seconds(), t3.Sub(t2).Seconds(), t3.Sub(t1).Seconds()}
}

func (c *CellCNN) PlaintextCircuitForwardOne() []complex128 {
	input := c.pcir.input
	filters := c.pcir.filters
	weights := c.pcir.weights
	actv := c.conv1d.PlainForwardCircuit(input, filters, c.cnnSettings)
	pred, u := c.dense.PlainForwardCircuit(weights, actv, c.cnnSettings)
	c.pcir.actv = actv
	c.pcir.pred = pred
	c.pcir.u = u
	return pred
}

func (c *CellCNN) PlaintextCircuitBackwardOne(err0 []complex128) ([][]complex128, []complex128) {
	input := c.pcir.input
	weights := c.pcir.weights
	actv := c.pcir.actv
	u := c.pcir.u
	dweights, nextErr := c.dense.PlainBackwardCircuit(weights, actv, u, err0, c.cnnSettings)
	dfilters := c.conv1d.PlainBackwardCircuit(input, nextErr, c.cnnSettings)
	return dfilters, dweights
}

func (c *CellCNN) ComputeLossOne(
	pred *ckks.Ciphertext, labels float64,
) *ckks.Ciphertext {
	// use naive least square loss: L = \sum (pred_i - label_i)^2
	// d-L / d-pred_i = 2 (pred_i - label_i)

	// 1. prepare the plaintext labels
	nfilters := c.cnnSettings.Nfilters
	onehot := utils.Float64ToOneHotEncode(labels, nfilters, c.params, c.encoder)

	out1 := c.evaluator.SubNew(pred, onehot)

	return out1
}

func (c *CellCNN) BackwardOne(err *ckks.Ciphertext) []float64 {
	t1 := time.Now()
	dsErr, _ := c.dense.Backward(err, c.cnnSettings, c.params, c.evaluator, c.encoder, c.sk, c.lr)
	t2 := time.Since(t1).Seconds()
	c.conv1d.Backward(dsErr, c.cnnSettings, c.params, c.evaluator, c.encoder, c.lr)
	t3 := time.Since(t1).Seconds()
	return []float64{t3 - t2, t2, t3}
}

func (c *CellCNN) ForwardAndBackwardOne(
	input *ckks.Plaintext,
	wConv []*ckks.Ciphertext,
	wDense *ckks.Ciphertext,
	poolMask *ckks.Plaintext,
	maskMap map[int]*ckks.Plaintext,
) ([]float64, []float64) {
	pred, tf := c.ForwardOne(input, wConv, wDense, poolMask)
	fakeLabel := 0
	loss := c.ComputeLossOne(pred, float64(fakeLabel))
	tb := c.BackwardOne(loss)
	return tf, tb
}

func (c *CellCNN) Matrix2Plaintext(rawData *mat.Dense) *ckks.Plaintext {
	// shape of each input: 200 * 37 (ncells = 200, nmakers = 37)
	row, col := rawData.Dims()
	value := make([]complex128, c.params.Slots())
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			value[i*col+j] = complex(rawData.At(i, j), 0)
		}
	}
	return c.encoder.EncodeNTTAtLvlNew(c.params.MaxLevel(), value, c.params.LogSlots())
}

func (c *CellCNN) MatrixTranspose2Plaintext(rawData *mat.Dense) *ckks.Plaintext {
	// shape of each input: 200 * 37 (ncells = 200, nmakers = 37)
	row, col := rawData.Dims()
	value := make([]complex128, c.params.Slots())
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			value[i+j*row] = complex(rawData.At(i, j), 0)
		}
	}
	return c.encoder.EncodeNTTAtLvlNew(c.params.MaxLevel(), value, c.params.LogSlots())
}

func (c *CellCNN) Batch2PlainSlice(inputs []*mat.Dense) []*ckks.Plaintext {
	result := make([]*ckks.Plaintext, 0)
	for _, each := range inputs {
		result = append(result, c.Matrix2Plaintext(each))
	}
	return result
}

func (c *CellCNN) ComputeScaledGradientWithMomentum(
	grad *Gradients,
	sts *utils.CellCnnSettings, params ckks.Parameters,
	eval ckks.Evaluator, encoder ckks.Encoder, momentum float64,
) *Gradients {
	Fgrad := c.conv1d.ComputeScaledGradientWithMomentum(grad.filters, sts, params, eval, encoder, momentum)
	Cgrad := c.dense.ComputeScaledGradientWithMomentum(grad.dense, sts, params, eval, encoder, momentum)
	return &Gradients{Fgrad, Cgrad}
}
