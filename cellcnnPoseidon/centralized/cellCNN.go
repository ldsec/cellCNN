package centralized

import (
	"fmt"
	"time"

	"github.com/ldsec/cellCNN/cellcnnPoseidon/layers"
	"github.com/ldsec/cellCNN/cellcnnPoseidon/utils"
	"github.com/ldsec/lattigo/v2/ckks"
	"gonum.org/v1/gonum/mat"
)

type Gradients struct {
	filters []*ckks.Ciphertext
	dense   *ckks.Ciphertext
}

func (g *Gradients) NewGradient(data [][]byte) {
	g.filters = make([]*ckks.Ciphertext, len(data)-1)
	for i, each := range data[:len(data)-1] {
		g.filters[i] = new(ckks.Ciphertext)
		if err := g.filters[i].UnmarshalBinary(each); err != nil {
			panic("fail to unmarshall Gradients")
		}
	}
	g.dense = new(ckks.Ciphertext)
	if err := g.dense.UnmarshalBinary(data[len(data)-1]); err != nil {
		panic("fail to unmarshall conv filter weights")
	}
}

// aggregate data to self
func (g *Gradients) Aggregate(data [][]byte, eval ckks.Evaluator) {
	// tmpFilters = make([]*ckks.Ciphertext, len(data)-1)
	for i, each := range data[:len(data)-1] {
		tmpFilter := new(ckks.Ciphertext)
		if err := tmpFilter.UnmarshalBinary(each); err != nil {
			panic("fail to unmarshall Gradients Aggregate")
		}
		eval.Add(g.filters[i], tmpFilter, g.filters[i])
	}
	tmpDense := new(ckks.Ciphertext)
	if err := tmpDense.UnmarshalBinary(data[len(data)-1]); err != nil {
		panic("fail to unmarshall conv filter weights")
	}
	eval.Add(g.dense, tmpDense, g.dense)
}

// aggregate data to self
func (g *Gradients) AggregateCt(filters []*ckks.Ciphertext, dense *ckks.Ciphertext, eval ckks.Evaluator) {
	// tmpFilters = make([]*ckks.Ciphertext, len(data)-1)
	for i, each := range filters {
		eval.Add(g.filters[i], each, g.filters[i])
	}
	eval.Add(g.dense, dense, g.dense)
}

// bootstrap
func (g *Gradients) Bootstrapping(encoder ckks.Encoder, params *ckks.Parameters, sk *ckks.SecretKey) {
	ect := ckks.NewEncryptorFromSk(params, sk)
	dct := ckks.NewDecryptor(params, sk)

	// re-encrypt filters
	for i, each := range g.filters {
		plain := encoder.Decode(dct.DecryptNew(each), params.LogSlots())
		replain := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), plain, params.LogSlots())
		g.filters[i] = ect.EncryptNew(replain)
	}

	// re-encrypt dense
	plain := encoder.Decode(dct.DecryptNew(g.dense), params.LogSlots())
	replain := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), plain, params.LogSlots())
	g.dense = ect.EncryptNew(replain)
}

func (g *Gradients) Marshall() [][]byte {
	res := make([][]byte, len(g.filters)+1)
	var err error = nil
	for i, each := range g.filters {
		res[i], err = each.MarshalBinary()
		if err != nil {
			panic("err in marshall Gradients")
		}
	}
	res[len(res)-1], err = g.dense.MarshalBinary()
	if err != nil {
		panic("err in marshall Gradients")
	}
	return res
}

func (g *Gradients) Unmarshall(data [][]byte) []*ckks.Ciphertext {
	res := make([]*ckks.Ciphertext, len(data))
	for i, each := range data {
		res[i] = new(ckks.Ciphertext)
		if err := res[i].UnmarshalBinary(each); err != nil {
			panic("fail to unmarshall Gradients")
		}
	}
	return res
}

func (g *Gradients) GetPlaintext(idx int, inds []int, params *ckks.Parameters, encoder ckks.Encoder, decryptor ckks.Decryptor) []complex128 {
	var ct *ckks.Ciphertext
	if idx < len(g.filters) {
		ct = g.filters[idx]
	} else {
		ct = g.dense
	}
	plaintext := encoder.Decode(decryptor.DecryptNew(ct), params.LogSlots())
	res := make([]complex128, len(inds))
	for i, each := range inds {
		res[i] = plaintext[each]
	}
	return res
}

type CellCNN struct {
	// network settings
	cnnSettings *utils.CellCnnSettings
	// crypto settings
	params    *ckks.Parameters
	relikey   *ckks.RelinearizationKey
	encoder   ckks.Encoder
	encryptor ckks.Encryptor
	// layers
	conv1d *layers.Conv1D
	// pooling *layers.Pool
	dense     *layers.Dense
	evaluator ckks.Evaluator
	btp       *ckks.Bootstrapper
	sk        *ckks.SecretKey
	// evaluator chan ckks.Evaluator
	pcir     *PlainCircuit
	momentum float64
	lr       float64
	// for debug
	// decryptor ckks.Decryptor

	// activation function settings
	// other attr
	// poolingMask  *ckks.Plaintext
	// denseMaskMap map[int]*ckks.Plaintext
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

// func (c *CellCNN) GetGradients() ([]*ckks.Ciphertext, *ckks.Ciphertext) {

// }

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

func NewCellCNN(
	// sts *utils.CellCnnSettings, params *ckks.Parameters, rlk *ckks.RelinearizationKey,
	// encoder ckks.Encoder, encryptor ckks.Encryptor,
	sts *utils.CellCnnSettings, cryptoParams *utils.CryptoParams, momentum, lr float64,
) *CellCNN {

	model := &CellCNN{
		cnnSettings: sts,
		params:      cryptoParams.Params,
		relikey:     cryptoParams.Rlk,
		encoder:     cryptoParams.GetEncoder(),
		encryptor:   cryptoParams.GetEncryptor(),
		momentum:    momentum,
		lr:          lr,
	}

	// if model.momentum {
	// 	model.SetMomentum()
	// }

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

func (c *CellCNN) WithSk(sk *ckks.SecretKey) {
	c.sk = sk
}

func (c *CellCNN) WithDiagM(diagM *ckks.PtDiagMatrix) {
	c.dense.WithDiagM(diagM)
}

// func (c *CellCNN) SetMomentum() {
// 	if c.conv1d != nil {
// 		c.conv1d.SetMomentum()
// 	}
// 	if c.dense != nil {
// 		c.dense.SetMomentum()
// 	}
// }

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
	fmt.Println("check params: nfilters, nmakers, nclasses, ncells", nfilters, nmakers, nclasses, ncells)
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

	fmt.Println("len of pConv, pDense:", len(pConv), len(pDense))

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
	// c.pooling = layers.NewPool(ncells, nmakers, nfilters)
	c.dense = layers.NewDense(wDense, c.momentum)
	c.dense.WithEncoder(encoder)
	fmt.Printf("==> CellCNN successfully initializing weights\n")
	return cm, dm
}

func (c *CellCNN) InitEvaluator(
	// kgen ckks.KeyGenerator,
	// sk *ckks.SecretKey,
	// encoder ckks.Encoder,
	// params *ckks.Parameters,
	cryptoParams *utils.CryptoParams,
	maxM1N2Ratio float64,
) ckks.Evaluator {

	kgen := cryptoParams.Kgen()
	encoder := c.encoder
	params := cryptoParams.Params
	sk := cryptoParams.Sk

	t1 := time.Now()

	Cinds := c.conv1d.InitRotationInds(c.cnnSettings, kgen)
	Dinds := c.dense.InitRotationInds(c.cnnSettings, kgen, c.params, encoder, maxM1N2Ratio)
	Rinds := utils.ClearRotInds(append(Cinds, Dinds...), params.Slots())

	rks := kgen.GenRotationKeysForRotations(Rinds, false, sk)

	t2 := time.Since(t1).Seconds()

	c.evaluator = ckks.NewEvaluator(c.params, ckks.EvaluationKey{Rlk: c.relikey, Rtks: rks})

	// cryptoParams.SetEvaluator(c.evaluator)

	fmt.Printf("==> CellCNN successfully creating evaluator\n")
	fmt.Printf("    with %v to generate rotation keys\n", t2)
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

	// nmakers := c.cnnSettings.Nmakers
	// nfilters := c.cnnSettings.Nfilters
	// ncells := c.cnnSettings.Ncells
	// nclasses := c.cnnSettings.Nclasses

	if poolMask == nil {

		// fmt.Println("got nil mask")
		// initialize masks required
		// conv1d left most mask
		LeftMostMask := utils.GenSliceWithOneAt(c.params.Slots(), []int{0})
		// LeftMostMask := make([]complex128, c.params.Slots())
		// LeftMostMask[0] = complex(float64(1), 0)
		poolMask = c.encoder.EncodeNTTAtLvlNew(c.params.MaxLevel(), LeftMostMask, c.params.LogSlots())

		// fmt.Println("got nil map")
		// dense maskMap to collect all results into one ciphertext
		// maskMap = make(map[int]*ckks.Plaintext)
		// for i := 0; i < nclasses; i++ {
		// 	maskMap[i*(nfilters-1)] = func() *ckks.Plaintext {
		// 		tmpMask := make([]complex128, c.params.Slots())
		// 		// fmt.Println("making maskMap: ", i*(nfilters-1))
		// 		tmpMask[i] = complex(float64(1), 0)
		// 		return c.encoder.EncodeNTTAtLvlNew(c.params.MaxLevel(), tmpMask, c.params.LogSlots())
		// 	}()
		// }

	}

	t1 := time.Now()
	out1 := c.conv1d.Forward(input, wConv, c.cnnSettings, c.evaluator, c.params, poolMask)
	// t2 := time.Now()
	// out2 := c.pooling.Forward(out1, c.evaluator, c.rotKeys, c.poolingMask, c.params.Slots())
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

	// put one hot in theta * nfilters place

	// 1. prepare the plaintext labels
	nfilters := c.cnnSettings.Nfilters
	onehot := utils.Float64ToOneHotEncode(labels, nfilters, c.params, c.encoder)

	out1 := c.evaluator.SubNew(pred, onehot)

	// out2 := c.evaluator.MultByConstNew(out1, 2)

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

// func (c *CellCNN) MockTrain(niter int, trainSet common.CnnDataset, batchSize int) []float64 {
// 	X := trainSet.X
// 	y := trainSet.Y
// 	tt := make([]float64, 3)
// 	for i := 1; i <= niter; i++ {
// 		// make a new batch
// 		newBatch := make([]*mat.Dense, batchSize)
// 		newBatchLabels := make([]float64, batchSize)
// 		for j := 0; j < len(newBatch); j++ {
// 			randi := rand.Intn(len(X))
// 			newBatch[j] = X[randi]
// 			newBatchLabels[j] = y[randi]
// 		}
// 		plaintextSlice := c.Batch2PlainSlice(newBatch)
// 		_, avgTime := c.ForwardBatch(plaintextSlice, i)
// 		for j := 0; j < len(tt); j++ {
// 			tt[j] = (tt[j]*float64(i-1) + avgTime[j]) / float64(i)
// 		}
// 		fmt.Printf(
// 			"=>Iter: %v |< Current: conv: %v | dense: %v | sum: %v >|< AVG: conv %v | dense: %v | sum: %v\n",
// 			i, avgTime[0], avgTime[1], avgTime[2], tt[0], tt[1], tt[2],
// 		)
// 	}
// 	return tt
// }

// func CompareTwoNetForward(
// 	eNet *CellCNN, pNet *PlainNet, cw, dw *mat.Dense,
// 	trainSet common.CnnDataset, niter int, batchSize int,
// 	decryptor ckks.Decryptor, encoder ckks.Encoder, params *ckks.Parameters,
// ) {
// 	X := trainSet.X
// 	y := trainSet.Y
// 	collector := 0.0
// 	amount := batchSize * niter * pNet.nclasses
// 	for i := 1; i <= niter; i++ {
// 		// make a new batch
// 		newBatch := make([]*mat.Dense, batchSize)
// 		newBatchLabels := make([]float64, batchSize)
// 		for j := 0; j < len(newBatch); j++ {
// 			randi := rand.Intn(len(X))
// 			newBatch[j] = X[randi]
// 			newBatchLabels[j] = y[randi]
// 		}
// 		// forward in encrypted net
// 		plaintextSlice := eNet.Batch2PlainSlice(newBatch)
// 		eo, _ := eNet.ForwardBatch(plaintextSlice, i)

// 		// forward in plain net
// 		po := pNet.ForwardBatch(newBatch, cw, dw)

// 		d1, d2 := po.Dims()
// 		fmt.Printf("Dims of plain net output: %v, %v\n", d1, d2)

// 		// Decrypt and compare pred
// 		fmt.Printf("Result for iteration %v\n", i)
// 		for j, each := range eo {
// 			predE := encoder.Decode(decryptor.DecryptNew(each), params.LogSlots())
// 			info := fmt.Sprintf("--> ID: %v", j)
// 			for c := 0; c < pNet.nclasses; c++ {
// 				er := real(predE[c*pNet.nfilters]) - po.At(j, c)
// 				collector += math.Pow(er, 2)
// 				info += fmt.Sprintf(
// 					"| Class %v E(%v) P(%v) | ", c, predE[c*pNet.nfilters], po.At(j, c),
// 				)
// 			}
// 			info += "\n"
// 			fmt.Printf(info)
// 		}
// 	}
// 	fmt.Printf(
// 		"Average error on one class: %v (tested over %v iterations with batchsize %v)",
// 		math.Pow(collector/float64(amount), 0.5), niter, batchSize,
// 	)
// }

// func CompareTwoNetBackward(
// 	eNet *CellCNN, pNet *PlainNet, cw, dw *mat.Dense,
// 	trainSet common.CnnDataset, niter int, batchSize int,
// 	decryptor ckks.Decryptor, encoder ckks.Encoder, params *ckks.Parameters,
// ) {
// 	X := trainSet.X
// 	y := trainSet.Y
// 	collector := 0.0
// 	// amount := batchSize * niter * pNet.nclasses
// 	for i := 1; i <= niter; i++ {
// 		// make a new batch
// 		newBatch := make([]*mat.Dense, batchSize)
// 		newBatchLabels := make([]float64, batchSize)
// 		for j := 0; j < len(newBatch); j++ {
// 			randi := rand.Intn(len(X))
// 			newBatch[j] = X[randi]
// 			newBatchLabels[j] = y[randi]
// 		}
// 		// forward in encrypted net
// 		plaintextSlice := eNet.Batch2PlainSlice(newBatch)
// 		eo, _ := eNet.ForwardBatch(plaintextSlice, i)

// 		// forward in plain net
// 		po := pNet.ForwardBatch(newBatch, cw, dw)

// 		// Decrypt and compare pred
// 		fmt.Printf("Result for iteration %v\n", i)
// 		for j, each := range eo {
// 			predE := encoder.Decode(decryptor.DecryptNew(each), params.LogSlots())
// 			info := fmt.Sprintf("--> ID: %v", j)
// 			for c := 0; c < pNet.nclasses; c++ {
// 				er := real(predE[c*pNet.nfilters]) - po.At(j, c)
// 				collector += math.Pow(er, 2)
// 				info += fmt.Sprintf(
// 					"| Class %v E(%v) P(%v) | ", c, predE[c*pNet.nfilters], po.At(j, c),
// 				)
// 			}
// 			info += "\n"
// 			fmt.Printf(info)
// 		}

// 		// copmute loss function
// 		errE := eNet.ComputeLossOne(eo[0], newBatchLabels[0])
// 		tmpLabel := make([]float64, pNet.nclasses)
// 		tmpLabel[int(newBatchLabels[0])] = 1
// 		labelsDense := mat.NewDense(1, pNet.nclasses, tmpLabel)
// 		errP := mat.NewDense(1, pNet.nclasses, nil)
// 		errP.Sub(po, labelsDense)

// 		// backward
// 		pconv, pdense := pNet.Backward(errP, eNet.lr, eNet.momentum)
// 		eNet.BackwardOne(errE)
// 		// eNet.Step(lr)

// 		// compare the accuracy and err
// 		// pconv := pNet.conv.GetWeights()
// 		// pdense := pNet.dense.GetWeights()

// 		econv := eNet.conv1d.GetGradient()
// 		edense := eNet.dense.GetGradient()

// 		decconv := make([][]complex128, pNet.nfilters)
// 		for i := range decconv {
// 			decconv[i] = encoder.Decode(decryptor.DecryptNew(econv[i]), params.LogSlots())
// 		}
// 		decdense := encoder.Decode(decryptor.DecryptNew(edense), params.LogSlots())

// 		Dmean, Dmse := utils.CompareDenseWeights(decdense, pdense, pNet.nfilters, pNet.nclasses)
// 		Cmean, Cmsd := utils.CompareConv1dWeights(decconv, pconv, pNet.nmakers, pNet.nfilters)

// 		fmt.Printf("\n######### In iteration <%v> Backward weights accuracay on each slot: ##########", i)
// 		fmt.Printf(
// 			"Conv1D: < mean: %v, mse: %v> || Dense: <mean: %v, mse: %v>\n\n",
// 			Cmean, Cmsd, Dmean, Dmse,
// 		)
// 	}
// }

// func (c *CellCNN) MockForwardBackward(niter int, trainSet common.CnnDataset, batchSize int) []float64 {
// 	X := trainSet.X
// 	y := trainSet.Y
// 	tt := make([]float64, 3)
// 	for i := 1; i <= niter; i++ {
// 		// make a new batch
// 		newBatch := make([]*mat.Dense, batchSize)
// 		newBatchLabels := make([]float64, batchSize)
// 		for j := 0; j < len(newBatch); j++ {
// 			randi := rand.Intn(len(X))
// 			newBatch[j] = X[randi]
// 			newBatchLabels[j] = y[randi]
// 		}
// 		plaintextSlice := c.Batch2PlainSlice(newBatch)

// 		output, avgTime := c.ForwardBatch(plaintextSlice, i)

// 		for j := 0; j < len(tt); j++ {
// 			tt[j] = (tt[j]*float64(i-1) + avgTime[j]) / float64(i)
// 		}
// 		fmt.Printf(
// 			"=>Iter: %v |< Current: conv: %v | dense: %v | sum: %v >|< AVG: conv %v | dense: %v | sum: %v\n",
// 			i, avgTime[0], avgTime[1], avgTime[2], tt[0], tt[1], tt[2],
// 		)
// 	}
// 	return tt
// }
