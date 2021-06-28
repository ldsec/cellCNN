package decentralized_test

import (
	"fmt"
	"github.com/ldsec/cellCNN/cellCNN_optimized"
	"github.com/ldsec/cellCNN/cellCNN_optimized/decentralized"
	"github.com/stretchr/testify/require"
	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"
	"testing"

	"go.dedis.ch/kyber/v3/group/edwards25519"
)

const PathCryptoFiles = "secrets/paramsCKKS"

// Suite is the type of keys used to secure communication in Onet
var Suite = edwards25519.NewBlakeSHA256Ed25519()

func TestRegEncryptedTraining(t *testing.T) {
	log.SetDebugVisible(2)

	t.Run("CellCNN", genTest(
		"cellCNN",           //protoID
		"../../normalized/",   // datapath
		3,                    // hosts
		true,              // trainPlain
		true,          // trainEncrypted
		true,        	// deterministic
		150,                 // Number of epochs
		true,                // debug
	))
}

func genTest(
	protoID, path string,
	hosts int,
	trainPlain, trainEncrypted, deterministic bool,
	epoch int,
	debug bool,
) func(*testing.T) {
	protoName := "CellCNN_optimized/" + protoID

	return func(t *testing.T) {
		log.SetDebugVisible(2)
		local := onet.NewLocalTest(Suite)
		defer local.CloseAll()

		servers, _, tree := local.GenTree(hosts, true)

		params := cellCNN.GenParams()

		cryptoParamsList := cellCNN.ReadOrGenerateCryptoParams(hosts, &params, PathCryptoFiles)
		require.NotNil(t, cryptoParamsList)

		// 1) Load Data
		log.Lvl2("Loading data...")
		XTrain, YTrain := cellCNN.LoadTrainDataFrom(path, cellCNN.Samples, cellCNN.Cells, cellCNN.Features)
		log.Lvl2("Done")

		localSamples := cellCNN.Samples / hosts // splits the data set

		for i, s := range servers {
			var XTrainS, YTrainS []*cellCNN.Matrix

			// Splits the data set
			if i-1 == hosts {
				XTrainS = XTrain[i*localSamples:]
				YTrainS = YTrain[i*localSamples:]

				localSamples = (cellCNN.Samples / hosts) + (cellCNN.Samples % hosts)

			} else {
				XTrainS = XTrain[i*localSamples : (i+1)*localSamples]
				YTrainS = YTrain[i*localSamples : (i+1)*localSamples]
			}

			_, err := s.ProtocolRegister(protoName, func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
				pi, err := decentralized.NewTrainingProtocol(tni)
				if err != nil {
					return nil, err
				}
				protocol := pi.(*decentralized.TrainingProtocol)

				vars := decentralized.InitCellCNNVars{
					TrainPlain:     trainPlain,
					TrainEncrypted: trainEncrypted,
					Deterministic:  deterministic,
					MaxIterations:  epoch,
					LocalSamples:   localSamples,
					Debug:          debug,
				}
				protocol.InitVars(cryptoParamsList[tni.Index()], &params, vars)

				protocol.XTrain = XTrainS
				protocol.YTrain = YTrainS

				return protocol, err
			})
			require.NoError(t, err)
		}

		rootInstance, err := local.CreateProtocol(protoName, tree)
		require.NoError(t, err)
		protocol := rootInstance.(*decentralized.TrainingProtocol)

		feedback := protocol.FeedbackChannel
		if err := protocol.Start(); err != nil {
			log.Panic(fmt.Errorf("start protocol: %v", err))
		}
		_ = <-feedback
	}
}