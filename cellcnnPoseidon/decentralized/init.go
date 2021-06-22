package decentralized

import (
	"github.com/ldsec/cellCNN/cellcnnPoseidon/centralized"
	"github.com/ldsec/cellCNN/cellcnnPoseidon/utils"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
)

// Params customized crypto parameters for secure bootstrapping.
// fresh ciphertext start a level 9. bootstrapping at level 2.
func Params() ckks.Parameters {
	pl := ckks.ParametersLiteral{
		LogN:     15,
		LogSlots: 14,
		LogQ:     []int{60, 60, 60, 52, 52, 52, 52, 52, 52, 52}, //60*3 + 52*7 = 544
		LogP:     []int{61, 61, 61},                             //183
		Scale:    1 << 52,
		Sigma:    rlwe.DefaultSigma,
	}
	params, err := ckks.NewParametersFromLiteral(pl)
	if err != nil {
		panic(err)
	}
	return params
}

// CustomizedParams return an object with decentralized.Params()
func CustomizedParams() *utils.CryptoParams {
	params := Params()

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	pk := kgen.GenPublicKey(sk)
	rlk := kgen.GenRelinearizationKey(sk)

	return utils.NewCryptoParams(params, sk, nil, pk, rlk)
}

// CustomizedNetworkKeysList return a slice of crypto-params object for federate learning use.
func CustomizedNetworkKeysList(nbrNodes int) []*utils.CryptoParams {
	return utils.NewCryptoParamsForNetwork(Params(), nbrNodes)
}

// Init initializes variables for cellCNN
func (p *NNEncryptedProtocol) Init(version string, crypto *utils.CryptoParams) error {

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

	p.evaluator = eval

	// use sk for bootstrapping
	p.model.WithSk(p.CryptoParams.AggregateSk)

	return nil
}
