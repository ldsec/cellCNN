package centralized

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/ldsec/cellCNN/cellCNN_clear/protocols/common"
	"github.com/ldsec/cellCNN/semester_project_shufan/layers"
	"github.com/ldsec/cellCNN/semester_project_shufan/utils"
	"github.com/ldsec/lattigo/v2/ckks"
	"gonum.org/v1/gonum/mat"
)

type CellCNN struct {
	// network settings
	cnnSettings *layers.CellCnnSettings
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
	pcir *PlainCircuit
	// for debug
	// decryptor ckks.Decryptor

	// activation function settings
	// other attr
	// poolingMask  *ckks.Plaintext
	// denseMaskMap map[int]*ckks.Plaintext
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

func NewCellCNN(
	sts *layers.CellCnnSettings, params *ckks.Parameters, rlk *ckks.RelinearizationKey,
	encoder ckks.Encoder, encryptor ckks.Encryptor,
) *CellCNN {

	model := &CellCNN{
		cnnSettings: sts,
		params:      params,
		relikey:     rlk,
		encoder:     encoder,
		encryptor:   encryptor,
	}
	return model
}

func (c *CellCNN) WithEvaluator(eval ckks.Evaluator) {
	c.evaluator = eval
}

func (c *CellCNN) SetMomentum() {
	if c.conv1d != nil {
		c.conv1d.SetMomentum()
	}
	if c.dense != nil {
		c.dense.SetMomentum()
	}
}

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

	c.conv1d = layers.NewConv1D(wConv1D)
	c.conv1d.WithEncoder(encoder)
	// c.pooling = layers.NewPool(ncells, nmakers, nfilters)
	c.dense = layers.NewDense(wDense)
	c.dense.WithEncoder(encoder)
	fmt.Printf("==> CellCNN successfully initializing weights\n")
	return cm, dm
}

func (c *CellCNN) InitEvaluator(
	kgen ckks.KeyGenerator,
	sk *ckks.SecretKey,
	encoder ckks.Encoder,
	params *ckks.Parameters,
	maxM1N2Ratio float64,
) {
	t1 := time.Now()
	Cinds := c.conv1d.InitRotationInds(c.cnnSettings, kgen)
	Dinds := c.dense.InitRotationInds(c.cnnSettings, kgen, c.params, encoder, maxM1N2Ratio)

	Rinds := utils.ClearRotInds(append(Cinds, Dinds...), params.Slots())

	rks := kgen.GenRotationKeysForRotations(Rinds, false, sk)

	t2 := time.Since(t1).Seconds()

	c.evaluator = ckks.NewEvaluator(c.params, ckks.EvaluationKey{Rlk: c.relikey, Rtks: rks})

	fmt.Printf("==> CellCNN successfully creating evaluator\n")
	fmt.Printf("    with %v to generate rotation keys\n", t2)
}

func (c *CellCNN) GenerateMaskMap() map[int]*ckks.Plaintext {
	nfilters := c.cnnSettings.Nfilters
	nclasses := c.cnnSettings.Nclasses
	// dense maskMap to collect all results into one ciphertext
	maskMap := make(map[int]*ckks.Plaintext)
	for i := 0; i < nclasses; i++ {
		maskMap[i*(nfilters-1)] = func() *ckks.Plaintext {
			tmpMask := make([]complex128, c.params.Slots())
			fmt.Println("making maskMap: ", i*(nfilters-1))
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
	maskMap map[int]*ckks.Plaintext,
) (*ckks.Ciphertext, []float64) {

	// nmakers := c.cnnSettings.Nmakers
	nfilters := c.cnnSettings.Nfilters
	// ncells := c.cnnSettings.Ncells
	nclasses := c.cnnSettings.Nclasses

	if poolMask == nil && maskMap == nil {

		fmt.Println("got nil mask")
		// initialize masks required
		// conv1d left most mask
		LeftMostMask := make([]complex128, c.params.Slots())
		LeftMostMask[0] = complex(float64(1), 0)
		poolMask = c.encoder.EncodeNTTAtLvlNew(c.params.MaxLevel(), LeftMostMask, c.params.LogSlots())

		fmt.Println("got nil map")
		// dense maskMap to collect all results into one ciphertext
		maskMap = make(map[int]*ckks.Plaintext)
		for i := 0; i < nclasses; i++ {
			maskMap[i*(nfilters-1)] = func() *ckks.Plaintext {
				tmpMask := make([]complex128, c.params.Slots())
				fmt.Println("making maskMap: ", i*(nfilters-1))
				tmpMask[i] = complex(float64(1), 0)
				return c.encoder.EncodeNTTAtLvlNew(c.params.MaxLevel(), tmpMask, c.params.LogSlots())
			}()
		}

	}

	t1 := time.Now()
	out1 := c.conv1d.Forward(input, wConv, c.cnnSettings, c.evaluator, c.params, poolMask)
	// t2 := time.Now()
	// out2 := c.pooling.Forward(out1, c.evaluator, c.rotKeys, c.poolingMask, c.params.Slots())
	t2 := time.Now()
	out2 := c.dense.Forward(out1, wDense, c.cnnSettings, c.evaluator, c.encoder, c.params, maskMap)
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

func (c *CellCNN) ForwardConv(
	input *ckks.Plaintext,
	wConv []*ckks.Ciphertext,
	wDense *ckks.Ciphertext,
	poolMask *ckks.Plaintext,
	maskMap map[int]*ckks.Plaintext,
) *ckks.Ciphertext {

	// nmakers := c.cnnSettings.Nmakers
	nfilters := c.cnnSettings.Nfilters
	// ncells := c.cnnSettings.Ncells
	nclasses := c.cnnSettings.Nclasses

	if poolMask == nil && maskMap == nil {

		ta := time.Now()

		fmt.Println("got nil mask")
		// initialize masks required
		// conv1d left most mask
		LeftMostMask := make([]complex128, c.params.Slots())
		LeftMostMask[0] = complex(float64(1), 0)
		poolMask = c.encoder.EncodeNTTAtLvlNew(c.params.MaxLevel(), LeftMostMask, c.params.LogSlots())

		fmt.Println("got nil map")
		// dense maskMap to collect all results into one ciphertext
		maskMap = make(map[int]*ckks.Plaintext)
		for i := 0; i < nclasses; i++ {
			maskMap[i*(nfilters-1)] = func() *ckks.Plaintext {
				tmpMask := make([]complex128, c.params.Slots())
				fmt.Println("making maskMap: ", i*(nfilters-1))
				tmpMask[i] = complex(float64(1), 0)
				return c.encoder.EncodeNTTAtLvlNew(c.params.MaxLevel(), tmpMask, c.params.LogSlots())
			}()
		}

		tb := time.Since(ta)

		fmt.Printf("Time consumed for generating rotation keys: %v\n", tb.Seconds())
	}

	out1 := c.conv1d.Forward(input, wConv, c.cnnSettings, c.evaluator, c.params, poolMask)

	return out1
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
	dsErr := c.dense.Backward(err, c.cnnSettings, c.params, c.evaluator, c.encoder, c.sk)
	t2 := time.Since(t1).Seconds()
	c.conv1d.Backward(dsErr, c.cnnSettings, c.params, c.evaluator, c.encoder)
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
	pred, tf := c.ForwardOne(input, wConv, wDense, poolMask, maskMap)
	fakeLabel := 0
	loss := c.ComputeLossOne(pred, float64(fakeLabel))
	tb := c.BackwardOne(loss)
	return tf, tb
}

func (c *CellCNN) Step(lr float64) bool {
	t1 := c.conv1d.Step(lr, 0, c.evaluator)
	t2 := c.dense.Step(lr, 0, c.evaluator)
	return t1 && t2
}

func (c *CellCNN) ForwardBatch(inputs []*ckks.Plaintext, j int) ([]*ckks.Ciphertext, []float64) {

	// nmakers := c.cnnSettings.Nmakers
	nfilters := c.cnnSettings.Nfilters
	// ncells := c.cnnSettings.Ncells
	nclasses := c.cnnSettings.Nclasses

	// initialize masks required
	// conv1d left most mask
	LeftMostMask := make([]complex128, c.params.Slots())
	LeftMostMask[0] = complex(float64(1), 0)
	poolMask := c.encoder.EncodeNTTAtLvlNew(c.params.MaxLevel(), LeftMostMask, c.params.LogSlots())

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

	// timer[0]:conv1d, [1]:dense, [2]:sum
	result := make([]*ckks.Ciphertext, 0)
	timer := make([]float64, 3)
	for id, sample := range inputs {
		output, t := c.ForwardOne(sample, nil, nil, poolMask, maskMap)
		result = append(result, output)
		for i := 0; i < len(timer); i++ {
			timer[i] += t[i]
		}
		utils.PrintTime(t, &id, "Forward One")
	}
	for i := 0; i < len(timer); i++ {
		timer[i] /= float64(len(inputs))
	}
	utils.PrintTime(timer, &j, "\nAVG Forward One in Batch")
	return result, timer
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

func (c *CellCNN) MockTrain(niter int, trainSet common.CnnDataset, batchSize int) []float64 {
	X := trainSet.X
	y := trainSet.Y
	tt := make([]float64, 3)
	for i := 1; i <= niter; i++ {
		// make a new batch
		newBatch := make([]*mat.Dense, batchSize)
		newBatchLabels := make([]float64, batchSize)
		for j := 0; j < len(newBatch); j++ {
			randi := rand.Intn(len(X))
			newBatch[j] = X[randi]
			newBatchLabels[j] = y[randi]
		}
		plaintextSlice := c.Batch2PlainSlice(newBatch)
		_, avgTime := c.ForwardBatch(plaintextSlice, i)
		for j := 0; j < len(tt); j++ {
			tt[j] = (tt[j]*float64(i-1) + avgTime[j]) / float64(i)
		}
		fmt.Printf(
			"=>Iter: %v |< Current: conv: %v | dense: %v | sum: %v >|< AVG: conv %v | dense: %v | sum: %v\n",
			i, avgTime[0], avgTime[1], avgTime[2], tt[0], tt[1], tt[2],
		)
	}
	return tt
}

func CompareTwoNetForward(
	eNet *CellCNN, pNet *PlainNet, cw, dw *mat.Dense,
	trainSet common.CnnDataset, niter int, batchSize int,
	decryptor ckks.Decryptor, encoder ckks.Encoder, params *ckks.Parameters,
) {
	X := trainSet.X
	y := trainSet.Y
	collector := 0.0
	amount := batchSize * niter * pNet.nclasses
	for i := 1; i <= niter; i++ {
		// make a new batch
		newBatch := make([]*mat.Dense, batchSize)
		newBatchLabels := make([]float64, batchSize)
		for j := 0; j < len(newBatch); j++ {
			randi := rand.Intn(len(X))
			newBatch[j] = X[randi]
			newBatchLabels[j] = y[randi]
		}
		// forward in encrypted net
		plaintextSlice := eNet.Batch2PlainSlice(newBatch)
		eo, _ := eNet.ForwardBatch(plaintextSlice, i)

		// forward in plain net
		po := pNet.ForwardBatch(newBatch, cw, dw)

		d1, d2 := po.Dims()
		fmt.Printf("Dims of plain net output: %v, %v\n", d1, d2)

		// Decrypt and compare pred
		fmt.Printf("Result for iteration %v\n", i)
		for j, each := range eo {
			predE := encoder.Decode(decryptor.DecryptNew(each), params.LogSlots())
			info := fmt.Sprintf("--> ID: %v", j)
			for c := 0; c < pNet.nclasses; c++ {
				er := real(predE[c*pNet.nfilters]) - po.At(j, c)
				collector += math.Pow(er, 2)
				info += fmt.Sprintf(
					"| Class %v E(%v) P(%v) | ", c, predE[c*pNet.nfilters], po.At(j, c),
				)
			}
			info += "\n"
			fmt.Printf(info)
		}
	}
	fmt.Printf(
		"Average error on one class: %v (tested over %v iterations with batchsize %v)",
		math.Pow(collector/float64(amount), 0.5), niter, batchSize,
	)
}

func CompareTwoNetBackward(
	eNet *CellCNN, pNet *PlainNet, cw, dw *mat.Dense,
	trainSet common.CnnDataset, niter int, batchSize int, lr float64,
	decryptor ckks.Decryptor, encoder ckks.Encoder, params *ckks.Parameters,
) {
	X := trainSet.X
	y := trainSet.Y
	collector := 0.0
	// amount := batchSize * niter * pNet.nclasses
	for i := 1; i <= niter; i++ {
		// make a new batch
		newBatch := make([]*mat.Dense, batchSize)
		newBatchLabels := make([]float64, batchSize)
		for j := 0; j < len(newBatch); j++ {
			randi := rand.Intn(len(X))
			newBatch[j] = X[randi]
			newBatchLabels[j] = y[randi]
		}
		// forward in encrypted net
		plaintextSlice := eNet.Batch2PlainSlice(newBatch)
		eo, _ := eNet.ForwardBatch(plaintextSlice, i)

		// forward in plain net
		po := pNet.ForwardBatch(newBatch, cw, dw)

		// Decrypt and compare pred
		fmt.Printf("Result for iteration %v\n", i)
		for j, each := range eo {
			predE := encoder.Decode(decryptor.DecryptNew(each), params.LogSlots())
			info := fmt.Sprintf("--> ID: %v", j)
			for c := 0; c < pNet.nclasses; c++ {
				er := real(predE[c*pNet.nfilters]) - po.At(j, c)
				collector += math.Pow(er, 2)
				info += fmt.Sprintf(
					"| Class %v E(%v) P(%v) | ", c, predE[c*pNet.nfilters], po.At(j, c),
				)
			}
			info += "\n"
			fmt.Printf(info)
		}

		// copmute loss function
		errE := eNet.ComputeLossOne(eo[0], newBatchLabels[0])
		tmpLabel := make([]float64, pNet.nclasses)
		tmpLabel[int(newBatchLabels[0])] = 1
		labelsDense := mat.NewDense(1, pNet.nclasses, tmpLabel)
		errP := mat.NewDense(1, pNet.nclasses, nil)
		errP.Sub(po, labelsDense)

		// backward
		pNet.Backward(errP, lr, 0)
		eNet.BackwardOne(errE)
		eNet.Step(lr)

		// compare the accuracy and err
		pconv := pNet.conv.GetWeights()
		pdense := pNet.dense.GetWeights()

		econv := eNet.conv1d.GetWeights()
		edense := eNet.dense.GetWeights()

		decconv := make([][]complex128, pNet.nfilters)
		for i := range decconv {
			decconv[i] = encoder.Decode(decryptor.DecryptNew(econv[i]), params.LogSlots())
		}
		decdense := encoder.Decode(decryptor.DecryptNew(edense), params.LogSlots())

		Dmean, Dmse := utils.CompareDenseWeights(decdense, pdense, pNet.nfilters, pNet.nclasses)
		Cmean, Cmsd := utils.CompareConv1dWeights(decconv, pconv, pNet.nmakers, pNet.nfilters)

		fmt.Printf("\n######### In iteration <%v> Backward weights accuracay on each slot: ##########", i)
		fmt.Printf(
			"Conv1D: < mean: %v, mse: %v> || Dense: <mean: %v, mse: %v>\n\n",
			Cmean, Cmsd, Dmean, Dmse,
		)
	}
}

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