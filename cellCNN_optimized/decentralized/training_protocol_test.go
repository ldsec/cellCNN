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

const PATH_CRYPTO_FILES = "secrets/paramsCKKS"

// Suite is the type of keys used to secure communication in Onet
var Suite = edwards25519.NewBlakeSHA256Ed25519()

func TestRegEncryptedTraining(t *testing.T) {
	log.SetDebugVisible(2)

	t.Run("CellCNN", genTest(
		"cellCNN",           //protoID
		"../../normalized/", // datapath
		3,                   // hosts
		false,               // trainEncrypted
		true,                // deterministic
		cellCNN.Samples,     //samples
		cellCNN.Cells,       //cells
		cellCNN.Features,    // features
		cellCNN.Filters,     // filters
		cellCNN.Classes,     // labels
		15,                  //epoch
		2000,                // party dataSize
		true,                // debug
	))
}

func genTest(
	protoID, path string,
	hosts int,
	trainEncrypted, deterministic bool,
	samples, cells, features, filters, classes int,
	epoch, partyDataSize int,
	debug bool,
) func(*testing.T) {
	protoName := "CellCNN_optimized/" + protoID

	return func(t *testing.T) {
		log.SetDebugVisible(2)
		local := onet.NewLocalTest(Suite)
		defer local.CloseAll()

		servers, _, tree := local.GenTree(hosts, true)

		params := cellCNN.GenParams()

		cryptoParamsList := cellCNN.ReadOrGenerateCryptoParams(hosts, &params, PATH_CRYPTO_FILES)
		require.NotNil(t, cryptoParamsList)

		// 1) Load Data
		log.Lvl2("Loading data...")
		XTrain, YTrain := cellCNN.LoadTrainDataFrom(path, samples, cells, features)
		log.Lvl2("Done")

		samplesPerHost := (samples / hosts)

		for i, s := range servers {
			var XTrainS, YTrainS []*cellCNN.Matrix

			if i-1 == hosts {
				XTrainS = XTrain[i*samplesPerHost:]
				YTrainS = YTrain[i*samplesPerHost:]

				samplesPerHost = (samples / hosts) + (samples % hosts)

			} else {
				XTrainS = XTrain[i*samplesPerHost : (i+1)*samplesPerHost]
				YTrainS = YTrain[i*samplesPerHost : (i+1)*samplesPerHost]
			}

			_, err := s.ProtocolRegister(protoName, func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
				pi, err := decentralized.NewTrainingProtocol(tni)
				if err != nil {
					return nil, err
				}
				protocol := pi.(*decentralized.TrainingProtocol)

				vars := decentralized.InitCellCNNVars{
					Path:           path,
					PartyDataSize:  partyDataSize / tni.Tree().Size(),
					TrainEncrypted: trainEncrypted,
					Deterministic:  deterministic,
					Epochs:         epoch * samples / cellCNN.BatchSize,
					Samples:        samplesPerHost,
					Cells:          cells,
					Features:       features,
					Filters:        filters,
					Classes:        classes,
					Debug:          debug,
				}
				protocol.InitVars(cryptoParamsList[tni.Index()], vars)

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
