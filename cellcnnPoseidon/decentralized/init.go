package decentralized

import (
	"github.com/ldsec/cellCNN/cellcnnPoseidon/centralized"
	"github.com/ldsec/cellCNN/cellcnnPoseidon/utils"
	"github.com/ldsec/lattigo/v2/ckks"
)

func CustomizedParams() *utils.CryptoParams {
	LogN := 15
	LogSlots := 14
	// logN=15: max 881 logQP
	LogModuli := ckks.LogModuli{
		LogQi: []int{60, 60, 60, 52, 52, 52, 52, 52, 52, 52}, //60*3 + 45*6 = 180 + 270 = 450
		LogPi: []int{61, 61, 61},                             //90
	}
	// sum of first 3 logQi == Scale +128
	Scale := float64(1 << 52)
	params, err := ckks.NewParametersFromLogModuli(LogN, &LogModuli)
	if err != nil {
		panic(err)
	}
	params.SetScale(Scale)
	params.SetLogSlots(LogSlots)

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	pk := kgen.GenPublicKey(sk)
	rlk := kgen.GenRelinearizationKey(sk)

	return utils.NewCryptoParams(params, sk, nil, pk, rlk)
}

func CustomizedNetworkKeysList(nbrNodes int) []*utils.CryptoParams {
	LogN := 15
	LogSlots := 14
	// logN=15: max 881 logQP
	LogModuli := ckks.LogModuli{
		LogQi: []int{60, 60, 60, 52, 52, 52, 52, 52, 52, 52}, //60*3 + 45*6 = 180 + 270 = 450
		LogPi: []int{61, 61, 61},                             //90
	}
	// sum of first 3 logQi == Scale +128
	Scale := float64(1 << 52)
	params, err := ckks.NewParametersFromLogModuli(LogN, &LogModuli)
	if err != nil {
		panic(err)
	}
	params.SetScale(Scale)
	params.SetLogSlots(LogSlots)

	return utils.NewCryptoParamsForNetwork(params, nbrNodes)
}

// Init initializes variables for cellCNN
func (p *NNEncryptedProtocol) Init(
	version string,
	crypto *utils.CryptoParams,
) error {

	p.CryptoParams = crypto

	p.encoder = p.CryptoParams.GetEncoder()

	p.Version = version

	p.CellCNNSettings = utils.NewCellCnnSettings(ncells, nmakers, nfilters, nclasses, approximationDegree, interval)

	// local learning rate
	p.LearningRate = learningRate / float64(HOSTS*nodeBatchSize)

	p.BatchSize = nodeBatchSize
	p.MaxIterations = maxIterations
	p.IterationNumber = 0

	p.ApproximationDegree = int(approximationDegree)
	p.Interval = []float64{-interval, interval}

	//only in root, and root will pass weights down the tree
	p.model = centralized.NewCellCNN(p.CellCNNSettings, p.CryptoParams, momentum, learningRate)
	p.model.InitWeights(nil, nil, nil, nil)
	eval, diagM := utils.InitEvaluator(p.CryptoParams, p.CellCNNSettings, maxM1N2Ratio)
	p.model.WithEvaluator(eval)
	p.model.WithDiagM(diagM)
	// eval := p.model.InitEvaluator(p.CryptoParams, maxM1N2Ratio)

	p.evaluator = eval

	// use sk for bootstrapping
	p.model.WithSk(p.CryptoParams.AggregateSk)

	return nil
}

// InitV2 initializes variables for neural network protocol with 2 layers
// func (p *NNEncryptedProtocol) InitV2(cryptoParams *libspindle.CryptoParams, initialGapSize,
// 	nInputLayer int, nHiddenLayer []int, nOutputLayer int, learningRate float64, nodeBatchSize, maxIterations int,
// 	approximationFunction leastsquares.ApproximationFunctionType, approximationDegree uint, interval []float64) error {

// 	p.CryptoParams = cryptoParams

// 	p.InitalGapSize = initialGapSize
// 	p.InitalGap = make([]float64, p.InitalGapSize)

// 	p.NInputs = nInputLayer
// 	p.NHiddens = nHiddenLayer
// 	p.NOutputs = nOutputLayer

// 	p.LearningRate = learningRate

// 	p.BatchSize = nodeBatchSize
// 	p.Epochs = maxIterations

// 	p.ApproximationDegree = int(approximationDegree)
// 	p.Interval = interval
// 	var err error
// 	if approximationFunction == leastsquares.Sigmoid {
// 		p.ApproximationFunction = "sigmoid"
// 		p.Coeffs, err = leastsquares.GetCoefficients(uint(p.ApproximationDegree), p.Interval[1], leastsquares.Sigmoid)
// 	} else if approximationFunction == leastsquares.Relu {
// 		p.ApproximationFunction = "relu"
// 		p.Coeffs, err = leastsquares.GetCoefficients(uint(p.ApproximationDegree), p.Interval[1], leastsquares.Relu)
// 	}
// 	if err != nil {
// 		return err
// 	}

// 	p.InitalGapSize = initialGapSize
// 	p.InitalGap = make([]float64, p.InitalGapSize)

// 	p.Weights = make([]libspindle.CipherVector, 3)

// 	p.HiddenActivations = make([]libspindle.CipherMatrix, len(p.NHiddens))
// 	p.HiddenBeforeActivations = make([]libspindle.CipherMatrix, len(p.NHiddens))

// 	for i := range p.HiddenActivations {
// 		p.HiddenActivations[i] = make(libspindle.CipherMatrix, p.BatchSize)
// 		p.HiddenBeforeActivations[i] = make(libspindle.CipherMatrix, p.BatchSize)
// 	}
// 	p.OutputActivations = make(libspindle.CipherVector, p.BatchSize)
// 	p.OutBeforeActivations = make(libspindle.CipherVector, p.BatchSize)

// 	//Parameters concerning only root
// 	learningRateVector := libspindle.VectorWith(p.CryptoParams.GetSlots(), learningRate)
// 	tmp, _ := libspindle.EncodeFloatVector(p.CryptoParams, learningRateVector, p.CryptoParams.Params.MaxLevel(), p.CryptoParams.Params.Scale())
// 	p.LearningRatePacked = tmp[0]
// 	//Root initialization of weights then passing down the tree
// 	InputWeights := libspindle.Matrix(p.NInputs, p.NHiddens[0])
// 	HiddenWeights := libspindle.Matrix(p.NHiddens[0], p.NHiddens[1])
// 	OutputWeights := libspindle.Matrix(p.NHiddens[1], p.NOutputs)

// 	//initialize weights
// 	for i := 0; i < p.NInputs; i++ {
// 		for j := 0; j < p.NHiddens[0]; j++ {
// 			InputWeights[i][j] = libspindle.Random(-1, 1) / math.Sqrt(float64(p.NInputs))
// 		}
// 	}
// 	for i := 0; i < p.NHiddens[1]; i++ {
// 		for j := 0; j < p.NOutputs; j++ {
// 			OutputWeights[i][j] = libspindle.Random(-1, 1) / math.Sqrt(float64(p.NHiddens[1]))
// 		}
// 	}
// 	for i := 0; i < p.NHiddens[0]; i++ {
// 		for j := 0; j < p.NHiddens[1]; j++ {
// 			HiddenWeights[i][j] = libspindle.Random(-1, 1) / math.Sqrt(float64(p.NHiddens[0]))
// 		}
// 	}

// 	gapInput := make([]float64, 0)
// 	gapHidden := make([]float64, 0)
// 	totalGapBetweenInput := 0
// 	totalGapBetweenHidden := 0
// 	if p.NHiddens[0] > p.NInputs {
// 		totalGapBetweenInput = p.NHiddens[0] - p.NInputs //put this gap between rows
// 		gapInput = make([]float64, totalGapBetweenInput)
// 		p.NumColsPerCipher = int(math.Floor(float64(p.CryptoParams.GetSlots()-p.InitalGapSize) / float64(p.NHiddens[0])))
// 		p.BlockSize = p.NHiddens[0] //Block size is important for each node
// 	}
// 	if p.NHiddens[0] < p.NInputs {
// 		totalGapBetweenHidden = p.NInputs - p.NHiddens[0] //put this gap between rows
// 		gapHidden = make([]float64, totalGapBetweenHidden)
// 		p.NumColsPerCipher = int(math.Floor(float64(p.CryptoParams.GetSlots()-p.InitalGapSize) / float64(p.NInputs)))
// 		p.BlockSize = p.NInputs //Block size is important for each node
// 	}

// 	//Transpose input matrix to temp to get columns in rows packed
// 	InputWeightsT := libspindle.Transpose(InputWeights)
// 	//check biggest all matrix sizes with gaps, see if the biggest fits into one cipher, otherwise create multiple ciphers
// 	inSize := p.NInputs*p.NHiddens[0] + ((p.NHiddens[0] - 1) * totalGapBetweenInput)
// 	hidSize := p.NHiddens[0]*p.NHiddens[1] + ((p.NHiddens[0] - 1) * totalGapBetweenHidden)
// 	outSize := p.NHiddens[1] * p.NOutputs
// 	largestSize := inSize
// 	if hidSize >= largestSize && hidSize >= outSize {
// 		largestSize = hidSize
// 	}
// 	if outSize >= largestSize && outSize >= hidSize {
// 		largestSize = outSize
// 	}
// 	if largestSize <= p.CryptoParams.GetSlots() {
// 		p.NumColsPerCipher = p.NHiddens[0]
// 		p.NumTotalCiphers = 1
// 		p.NumColsLastCipher = p.NumColsPerCipher
// 	} else {
// 		fmt.Println("Largest Size is", largestSize)
// 		if math.Mod(float64(largestSize), float64(p.CryptoParams.GetSlots()-p.InitalGapSize)) == 0 {
// 			p.NumTotalCiphers = int(math.Floor(float64(p.NHiddens[0]) / float64(p.NumColsPerCipher)))
// 			p.NumColsLastCipher = p.NumColsPerCipher

// 		} else {
// 			p.NumTotalCiphers = int(math.Floor(float64(p.NHiddens[0])/float64(p.NumColsPerCipher))) + 1
// 			p.NumColsLastCipher = p.NHiddens[0] - ((p.NumTotalCiphers - 1) * p.NumColsPerCipher)
// 		}
// 	}
// 	fmt.Println("Num columns/rows per cipher is", p.NumColsPerCipher)
// 	fmt.Println("Num total cipher is", p.NumTotalCiphers)
// 	fmt.Println("Num last cipher is", p.NumColsLastCipher)

// 	if p.NumColsLastCipher == p.NumColsPerCipher {
// 		p.IsLastDifferent = false
// 	}
// 	//Vectorize "columns" and pack into multi cipher:
// 	p.Weights[0] = make([]*ckks.Ciphertext, p.NumTotalCiphers)
// 	ind := 0
// 	for k := 0; k < p.NumTotalCiphers-1; k++ {
// 		InputWeightsColumns := make([]float64, 0)
// 		InputWeightsColumns = append(InputWeightsColumns, p.InitalGap...)
// 		for j := 0; j < p.NumColsPerCipher; j++ {
// 			InputWeightsColumns = append(InputWeightsColumns, InputWeightsT[ind]...)
// 			InputWeightsColumns = append(InputWeightsColumns, gapInput...)
// 			ind = ind + 1
// 		}
// 		packed, _ := libspindle.EncodeFloatVector(p.CryptoParams, InputWeightsColumns, p.CryptoParams.Params.MaxLevel(), p.CryptoParams.Params.Scale())

// 		_ = p.CryptoParams.WithEncryptor(func(encryptor ckks.Encryptor) error {
// 			p.Weights[0][k] = encryptor.EncryptNew(packed[0])
// 			return nil
// 		})
// 	}
// 	//vectorize last one
// 	InputWeightsColumns := make([]float64, 0)
// 	InputWeightsColumns = append(InputWeightsColumns, p.InitalGap...)
// 	for j := 0; j < p.NumColsLastCipher; j++ {
// 		InputWeightsColumns = append(InputWeightsColumns, InputWeightsT[ind]...)
// 		InputWeightsColumns = append(InputWeightsColumns, gapInput...)
// 		ind = ind + 1
// 	}
// 	packed, _ := libspindle.EncodeFloatVector(p.CryptoParams, InputWeightsColumns, p.CryptoParams.Params.MaxLevel(), p.CryptoParams.Params.Scale())
// 	_ = p.CryptoParams.WithEncryptor(func(encryptor ckks.Encryptor) error {
// 		p.Weights[0][p.NumTotalCiphers-1] = encryptor.EncryptNew(packed[0])
// 		return nil
// 	})

// 	//Vectorize "rows" of hidden weights and encrypt
// 	p.Weights[1] = make([]*ckks.Ciphertext, p.NumTotalCiphers)
// 	ind = 0
// 	for k := 0; k < p.NumTotalCiphers-1; k++ {
// 		HiddenWeightRows := make([]float64, 0)
// 		HiddenWeightRows = append(HiddenWeightRows, p.InitalGap...)
// 		for j := 0; j < p.NumColsPerCipher; j++ {
// 			HiddenWeightRows = append(HiddenWeightRows, HiddenWeights[ind]...)
// 			HiddenWeightRows = append(HiddenWeightRows, gapHidden...)
// 			ind = ind + 1
// 		}
// 		packed, _ := libspindle.EncodeFloatVector(p.CryptoParams, HiddenWeightRows, p.CryptoParams.Params.MaxLevel(), p.CryptoParams.Params.Scale())
// 		_ = p.CryptoParams.WithEncryptor(func(encryptor ckks.Encryptor) error {
// 			p.Weights[1][k] = encryptor.EncryptNew(packed[0])
// 			return nil
// 		})
// 	}
// 	//vectorize last one
// 	HiddenWeightRows := make([]float64, 0)
// 	HiddenWeightRows = append(HiddenWeightRows, p.InitalGap...)
// 	for j := 0; j < p.NumColsLastCipher; j++ {
// 		HiddenWeightRows = append(HiddenWeightRows, HiddenWeights[ind]...)
// 		HiddenWeightRows = append(HiddenWeightRows, gapHidden...)
// 		ind = ind + 1
// 	}
// 	packed, _ = libspindle.EncodeFloatVector(p.CryptoParams, HiddenWeightRows, p.CryptoParams.Params.MaxLevel(), p.CryptoParams.Params.Scale())
// 	_ = p.CryptoParams.WithEncryptor(func(encryptor ckks.Encryptor) error {
// 		p.Weights[1][p.NumTotalCiphers-1] = encryptor.EncryptNew(packed[0])
// 		return nil
// 	})

// 	//Transpose output to temp to get columns in rows packed
// 	OutputWeightsT := libspindle.Transpose(OutputWeights)
// 	p.Weights[2] = make([]*ckks.Ciphertext, 1)
// 	ind = 0
// 	//vectorize output weights one-cipher (always fits into one cipher)
// 	OutputWeightsColumns := make([]float64, 0)
// 	OutputWeightsColumns = append(OutputWeightsColumns, p.InitalGap...)
// 	for j := 0; j < p.NOutputs; j++ {
// 		OutputWeightsColumns = append(OutputWeightsColumns, OutputWeightsT[ind]...)
// 		ind = ind + 1
// 	}
// 	packed, _ = libspindle.EncodeFloatVector(p.CryptoParams, OutputWeightsColumns, p.CryptoParams.Params.MaxLevel(), p.CryptoParams.Params.Scale())
// 	_ = p.CryptoParams.WithEncryptor(func(encryptor ckks.Encryptor) error {
// 		p.Weights[2][0] = encryptor.EncryptNew(packed[0])
// 		return nil
// 	})

// 	//Offline mask creation, masks should be in each node
// 	mask := make([]float64, p.CryptoParams.GetSlots())
// 	ind = p.InitalGapSize
// 	for i := 0; i < p.NumColsPerCipher; i++ {
// 		mask[ind] = 1
// 		ind += p.BlockSize
// 	}
// 	tmp, _ = libspindle.EncodeFloatVector(p.CryptoParams, mask, p.CryptoParams.Params.MaxLevel(), p.CryptoParams.Params.Scale())
// 	p.Mask = tmp[0]

// 	mask = make([]float64, p.CryptoParams.GetSlots())
// 	ind = p.InitalGapSize
// 	for i := 0; i < p.NumColsLastCipher; i++ {
// 		mask[ind] = 1
// 		ind += p.BlockSize
// 	}
// 	tmp, _ = libspindle.EncodeFloatVector(p.CryptoParams, mask, p.CryptoParams.Params.MaxLevel(), p.CryptoParams.Params.Scale())
// 	p.MaskLast = tmp[0]

// 	ind = p.InitalGapSize
// 	mask = make([]float64, p.CryptoParams.GetSlots())
// 	totalGapBetween := p.NHiddens[1]
// 	for i := 0; i < p.NOutputs; i++ {
// 		mask[ind] = 1
// 		ind += totalGapBetween
// 	}
// 	tmp, _ = libspindle.EncodeFloatVector(p.CryptoParams, mask, p.CryptoParams.Params.MaxLevel(), p.CryptoParams.Params.Scale())
// 	p.Mask2 = tmp[0]

// 	mask = make([]float64, p.CryptoParams.GetSlots())
// 	for i := 0; i < p.NHiddens[1]; i++ {
// 		mask[i+p.InitalGapSize] = 1
// 	}
// 	tmp, _ = libspindle.EncodeFloatVector(p.CryptoParams, mask, p.CryptoParams.Params.MaxLevel(), p.CryptoParams.Params.Scale())
// 	p.Mask3 = tmp[0]

// 	return nil
// }
