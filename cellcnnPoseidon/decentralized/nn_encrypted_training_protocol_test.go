package decentralized

import (
	"fmt"
	"testing"

	"github.com/ldsec/cellCNN/cellcnnPoseidon/utils"
	"github.com/stretchr/testify/require"
	"go.dedis.ch/onet/v3"
)

var cryptoParamsList []*utils.CryptoParams

var seed []byte

// func TestNNEncryptedTraining(t *testing.T) {
// 	log.SetDebugVisible(2)

// 	fmt.Println("start running")
// 	// t.Run("BCW", genTest(
// 	// 	"bcw",
// 	// 	// "v1",
// 	// 	// loaders.Pima{Path: "../../../data/bcw/bcw.csv"},
// 	// 	// []int{0, 1},
// 	// 	// protocols.NeuralNetwork,
// 	// 	// "v1", 16, []int{64}, 2,
// 	// 	// 0.3,
// 	// 	// 1, 2, 2,
// 	// 	// leastsquares.Relu, 3, []float64{-3.0, 3.0},
// 	// 	// false, true,
// 	// 	// []float64{0, 0, 0.7330729166666667, 0.6004190844616377, 0.7078063186047807, 0.6484448411952641, 0.8020537338833305},
// 	// ))
// 	fmt.Println("start running")
// }

func TestDemo(t *testing.T) {
	protoID := "BCW"
	protoName := "NNEncryptedTest/" + protoID

	local := onet.NewLocalTest(GetSuit())
	defer local.CloseAll()

	var err error
	// seed, err = libspindle.GenRandSeed()
	// require.NoError(t, err)

	// cryptoParamsList = nnenc.ReadOrGenerateCryptoParamsForNN(HOSTS, N, PATH_CRYPTO_FILES, nInputLayer, nOutputLayer, nHiddenLayer, false)
	// require.NotNil(t, cryptoParamsList)

	servers, _, tree := local.GenTree(HOSTS, true)
	for _, s := range servers {
		_, err := s.ProtocolRegister(protoName, func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
			pi, err := NewNNEncryptedProtocol(tni)
			if err != nil {
				return nil, err
			}
			protocol := pi.(*NNEncryptedProtocol)

			// 1. split the data for each server
			// load the dataset without split for test
			// path := "../data/cellCNN/normalized/"
			// trainData := common.LoadTrainDataFrom(path)
			// protocol.TrainSet = &trainData
			protocol.TrainSet = nil
			protocol.Sync = true

			// 2. init protocol training variables
			err = protocol.Init("v1")

			// // ##STEP 1: Split data
			// X, Y, maxIterations := spindleprotcommon.SplitDataForTest(trainType, spindleprotcommon.TrainData.X, spindleprotcommon.TrainData.Y, len(tni.Roster().List), tni.Index(),
			// 	numberOfDatasetUsed, numberOfLocalIteration, nodeBatchSize, false, 0.0, tni.IsRoot())
			// protocol.X = X
			// // for neural networks the labels are of the format 1 -> [0 1]
			// protocol.Y = libspindle.ConvertLabelsToNNFormat(Y, len(classes))

			// // ##STEP 2: Init protocol training variables
			// if version == "v1" {
			// err = protocol.Init(cryptoParamsList[tni.Index()], HOSTS, INITIAL_GAP_SIZE, version, nInputLayer, nHiddenLayer, nOutputLayer,
			// 		alpha, nodeBatchSize, maxIterations, approximationFunction, approximationDegree, interval)
			// 	if err != nil {
			// 		return nil, err
			// 	}
			// } else {
			// 	err = protocol.InitV2(cryptoParamsList[tni.Index()], INITIAL_GAP_SIZE, nInputLayer, nHiddenLayer, nOutputLayer,
			// 		alpha, nodeBatchSize, maxIterations, approximationFunction, approximationDegree, interval)
			// 	if err != nil {
			// 		return nil, err
			// 	}
			// }
			// protocol.Debug = DEBUG
			// protocol.Debug.ServerID = tni.ServerIdentity().String()
			return protocol, err
		})
		require.NoError(t, err)

		// _, err = s.ProtocolRegister(decentralized.BootstrapProtocolName, decentralized.NewBootstrapProtocolFunction(cryptoParamsList[sIdx], seed))
		// require.NoError(t, err)
	}

	fmt.Println("start running")
	// cryptoParamsList[0].SetDecryptors(cryptoParamsList[0].Params, cryptoParamsList[0].AggregateSk)
	wBytes, err := RunNeuralNetworkTest(local, nil, tree, true, true, protoName)

	fmt.Println(wBytes[0])
	// for i := range expectedStats {
	// 	require.InDelta(t, expectedStats[i], stats[i], 0.1)
	// }

	require.NoError(t, err)

}

func genTest(
	protoID string,
	// loader *common.CnnDataset,
	// classes []int,
	// trainType protocols.TrainingType,
	// version string,
	// version string, nInputLayer int, nHiddenLayer []int, nOutputLayer int,
	// alpha float64,
	// numberOfDatasetUsed, numberOfLocalIteration, nodeBatchSize int,
	// approximationFunction leastsquares.ApproximationFunctionType,
	// approximationDegree uint,
	// interval []float64,
	// normalize, standardize bool, expectedStats []float64,
) func(*testing.T) {
	protoName := "NNEncryptedTest/" + protoID
	return func(t *testing.T) {
		t.Skip()

		local := onet.NewLocalTest(GetSuit())
		defer local.CloseAll()

		var err error
		// seed, err = libspindle.GenRandSeed()
		// require.NoError(t, err)

		// cryptoParamsList = nnenc.ReadOrGenerateCryptoParamsForNN(HOSTS, N, PATH_CRYPTO_FILES, nInputLayer, nOutputLayer, nHiddenLayer, false)
		// require.NotNil(t, cryptoParamsList)

		servers, _, tree := local.GenTree(HOSTS, true)
		for _, s := range servers {
			_, err := s.ProtocolRegister(protoName, func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
				pi, err := NewNNEncryptedProtocol(tni)
				if err != nil {
					return nil, err
				}
				protocol := pi.(*NNEncryptedProtocol)

				// 1. split the data for each server
				// load the dataset without split for test
				// path := "../data/cellCNN/normalized/"
				// trainData := common.LoadTrainDataFrom(path)
				// protocol.TrainSet = &trainData
				protocol.TrainSet = nil
				protocol.Sync = true

				// 2. init protocol training variables
				err = protocol.Init("v1")

				// // ##STEP 1: Split data
				// X, Y, maxIterations := spindleprotcommon.SplitDataForTest(trainType, spindleprotcommon.TrainData.X, spindleprotcommon.TrainData.Y, len(tni.Roster().List), tni.Index(),
				// 	numberOfDatasetUsed, numberOfLocalIteration, nodeBatchSize, false, 0.0, tni.IsRoot())
				// protocol.X = X
				// // for neural networks the labels are of the format 1 -> [0 1]
				// protocol.Y = libspindle.ConvertLabelsToNNFormat(Y, len(classes))

				// // ##STEP 2: Init protocol training variables
				// if version == "v1" {
				// err = protocol.Init(cryptoParamsList[tni.Index()], HOSTS, INITIAL_GAP_SIZE, version, nInputLayer, nHiddenLayer, nOutputLayer,
				// 		alpha, nodeBatchSize, maxIterations, approximationFunction, approximationDegree, interval)
				// 	if err != nil {
				// 		return nil, err
				// 	}
				// } else {
				// 	err = protocol.InitV2(cryptoParamsList[tni.Index()], INITIAL_GAP_SIZE, nInputLayer, nHiddenLayer, nOutputLayer,
				// 		alpha, nodeBatchSize, maxIterations, approximationFunction, approximationDegree, interval)
				// 	if err != nil {
				// 		return nil, err
				// 	}
				// }
				// protocol.Debug = DEBUG
				// protocol.Debug.ServerID = tni.ServerIdentity().String()
				return protocol, err
			})
			require.NoError(t, err)

			// _, err = s.ProtocolRegister(decentralized.BootstrapProtocolName, decentralized.NewBootstrapProtocolFunction(cryptoParamsList[sIdx], seed))
			// require.NoError(t, err)
		}

		fmt.Println("start running")
		cryptoParamsList[0].SetDecryptors(cryptoParamsList[0].Params, cryptoParamsList[0].AggregateSk)
		wBytes, err := RunNeuralNetworkTest(local, nil, tree, true, true, protoName)

		fmt.Println(wBytes[0])
		// for i := range expectedStats {
		// 	require.InDelta(t, expectedStats[i], stats[i], 0.1)
		// }

		require.NoError(t, err)
	}
}
