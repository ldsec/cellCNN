package centralized

import (
	"math/rand"
	"time"

	"github.com/ldsec/cellCNN/cellcnnPoseidon/layers"
	"github.com/ldsec/cellCNN/cellcnnPoseidon/utils"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"gonum.org/v1/gonum/mat"
)

// PlainCircuit is a plaintext circuit only for debug
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

// NewPlainCircuit init a new circuit
func NewPlainCircuit(filters [][]complex128, weights []complex128, input []complex128) *PlainCircuit {
	return &PlainCircuit{
		filters: filters,
		weights: weights,
		input:   input,
	}
}

// CellCNN is an encrypted network that conduct forward and backward
// please set the sk if you want to use dummy bootstrapping in backward
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
	sk        *rlwe.SecretKey
	pcir      *PlainCircuit
	momentum  float64
	lr        float64
}

// GetEncoder for debug use, return the encoder
func (c *CellCNN) GetEncoder() ckks.Encoder {
	return c.encoder
}

// GetEvaluator for debug use, return the evaluator
func (c *CellCNN) GetEvaluator() ckks.Evaluator {
	return c.evaluator
}

// GetWeights return the weights
// the first n-1 ciphertexts are the filters
// the last ciphertext is the dense weights
func (c *CellCNN) GetWeights() []*ckks.Ciphertext {
	econv := c.conv1d.GetWeights()
	edense := c.dense.GetWeights()
	return append(econv, edense)
}

// GetGradients return the gradients as a new object
func (c *CellCNN) GetGradients() *Gradients {
	return &Gradients{c.conv1d.GetGradient(), c.dense.GetGradient()}

}

// NewCellCNN init a new Cell CNN with the settings
// the weights will not be initialized
// call InitWeights to init the weights
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

// UpdateWithGradients will update the weights of Cell CNN by g.
// w_{new} = w_{old} - g
func (c *CellCNN) UpdateWithGradients(g *Gradients) {
	c.conv1d.UpdateWithGradients(g.filters, c.evaluator)
	c.dense.UpdateWithGradients(g.dense, c.evaluator)
}

// GetWeightsBinary return the binary weights of Cell CNN.
// the first n-1 are filter weights
// the last one is dense weights
func (c *CellCNN) GetWeightsBinary() (data [][]byte) {
	filterData := c.conv1d.GetWeightsBinary()
	denseData := c.dense.GetWeightsBinary()
	data = append(filterData, denseData)
	return
}

// LoadWeightsBinary will load the weights according to data
func (c *CellCNN) LoadWeightsBinary(data [][]byte) {
	nfilters := len(data) - 1
	c.conv1d.LoadWeightsBinary(data[:nfilters])
	c.dense.LoadWeightsBinary(data[nfilters])
}

// GetGradient return the the graident
func (c *CellCNN) GetGradient() []*ckks.Ciphertext {
	filters := c.conv1d.GetGradient()
	dense := c.dense.GetGradient()
	return append(filters, dense)
}

// GetGradientBinary return the byte representation of the graident.
// the first n-1 is conv gradients,
// the last one is dense gradient
func (c *CellCNN) GetGradientBinary() [][]byte {
	dConv := c.conv1d.GetGradientBinary()
	dDense := c.dense.GetGradientBinary()
	return append(dConv, dDense)
}

// WithEvaluator for debug use, set the evaluator of CellCNN
func (c *CellCNN) WithEvaluator(eval ckks.Evaluator) {
	c.evaluator = eval
}

// WithSk set the Sk for dummy bootstrapping: re-encrypt
func (c *CellCNN) WithSk(sk *rlwe.SecretKey) {
	c.sk = sk
}

// WithDiagM set dense.diagM for debug use or used in decentralized settings
func (c *CellCNN) WithDiagM(diagM *ckks.PtDiagMatrix) {
	c.dense.WithDiagM(diagM)
}

// FisrtMomentum check if has first momentum
func (c *CellCNN) FisrtMomentum() bool {
	return c.conv1d.FirstMomentum() && c.dense.FirstMomentum()
}

// UpdateMomentum store the ciphertexts in grad as new momentum for next iteration use
func (c *CellCNN) UpdateMomentum(grad *Gradients) {
	c.conv1d.UpdateMomentum(utils.CopyCiphertextSlice(grad.filters))
	c.dense.UpdateMomentum(grad.dense.CopyNew())
}

// InitWeights init CellCNN and return the init plaintext weights (conv, dense) as mat.Dense.
// this is useful to initialize a cellCNNClear with same weights.
func (c *CellCNN) InitWeights(wConv1D []*ckks.Ciphertext, wDense *ckks.Ciphertext, pConv, pDense []complex128) (*mat.Dense, *mat.Dense) {
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

// InitEvaluator init the evaluators of Cell CNN
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

// ForwardOne forward only one sample (not a batch).
// return the ciphertext predition and the time consumed on (conv, dense, sum)
func (c *CellCNN) ForwardOne(input *ckks.Plaintext, wConv []*ckks.Ciphertext, wDense *ckks.Ciphertext) (*ckks.Ciphertext, []float64) {
	t1 := time.Now()
	out1 := c.conv1d.Forward(input, wConv, c.cnnSettings, c.evaluator, c.params)
	t2 := time.Now()
	out2 := c.dense.Forward(out1, wDense, c.cnnSettings, c.evaluator, c.encoder, c.params)
	t3 := time.Now()
	return out2, []float64{t2.Sub(t1).Seconds(), t3.Sub(t2).Seconds(), t3.Sub(t1).Seconds()}
}

// ComputeLossOne compute the mean square error.
// L = \sum (pred_i - label_i)^2.
// d-L / d-pred_i = 2 (pred_i - label_i)
func (c *CellCNN) ComputeLossOne(
	pred *ckks.Ciphertext, labels float64,
) *ckks.Ciphertext {
	// prepare the plaintext labels
	nfilters := c.cnnSettings.Nfilters
	onehot := utils.Float64ToOneHotEncode(labels, nfilters, c.params, c.encoder)

	out1 := c.evaluator.SubNew(pred, onehot)

	return out1
}

// BackwardOne backward only one sample according to input err.
// returns the time for backward in (conv, dense, sum)
func (c *CellCNN) BackwardOne(err *ckks.Ciphertext) []float64 {
	t1 := time.Now()
	dsErr, _ := c.dense.Backward(err, c.cnnSettings, c.params, c.evaluator, c.encoder, c.sk, c.lr)
	t2 := time.Since(t1).Seconds()
	c.conv1d.Backward(dsErr, c.cnnSettings, c.params, c.evaluator, c.encoder, c.lr)
	t3 := time.Since(t1).Seconds()
	return []float64{t3 - t2, t2, t3}
}

// PlaintextCircuitForwardOne for debug use, compute forward on plaintext circuit
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

// PlaintextCircuitBackwardOne for debug use, compute backward on plaintext circuit
func (c *CellCNN) PlaintextCircuitBackwardOne(err0 []complex128) ([][]complex128, []complex128) {
	input := c.pcir.input
	weights := c.pcir.weights
	actv := c.pcir.actv
	u := c.pcir.u
	dweights, nextErr := c.dense.PlainBackwardCircuit(weights, actv, u, err0, c.cnnSettings)
	dfilters := c.conv1d.PlainBackwardCircuit(input, nextErr, c.cnnSettings)
	return dfilters, dweights
}

// ForwardAndBackwardOne for debug only,
// it forward and backward only one sample,
// it use randomly generated fake label to compute loss.
func (c *CellCNN) ForwardAndBackwardOne(input *ckks.Plaintext, wConv []*ckks.Ciphertext, wDense *ckks.Ciphertext) ([]float64, []float64) {
	pred, tf := c.ForwardOne(input, wConv, wDense)
	fakeLabel := rand.Intn(2)
	loss := c.ComputeLossOne(pred, float64(fakeLabel))
	tb := c.BackwardOne(loss)
	return tf, tb
}

// Matrix2Plaintext row pack a matrix to a ckks plaintext
// useful for generating dataset
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

// MatrixTranspose2Plaintext column pack a matrix a ckks plaintext
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

// Batch2PlainSlice row pack a batch of matrices to a slice of plaintext
func (c *CellCNN) Batch2PlainSlice(inputs []*mat.Dense) []*ckks.Plaintext {
	result := make([]*ckks.Plaintext, 0)
	for _, each := range inputs {
		result = append(result, c.Matrix2Plaintext(each))
	}
	return result
}

// ComputeScaledGradientWithMomentum add the momentum to the scaled gradients
func (c *CellCNN) ComputeScaledGradientWithMomentum(
	grad *Gradients,
	sts *utils.CellCnnSettings, params ckks.Parameters,
	eval ckks.Evaluator, encoder ckks.Encoder, momentum float64,
) *Gradients {
	Fgrad := c.conv1d.ComputeScaledGradientWithMomentum(grad.filters, sts, params, eval, encoder, momentum)
	Cgrad := c.dense.ComputeScaledGradientWithMomentum(grad.dense, sts, params, eval, encoder, momentum)
	return &Gradients{Fgrad, Cgrad}
}
