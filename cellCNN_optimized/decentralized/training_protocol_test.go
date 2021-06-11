package decentralized_test

import (
	"fmt"
	"github.com/ldsec/cellCNN/cellCNN_optimized"
	"github.com/ldsec/cellCNN/cellCNN_optimized/decentralized"
	"github.com/stretchr/testify/require"
	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"
	"math/rand"
	"testing"
	"time"

	"go.dedis.ch/kyber/v3/group/edwards25519"
)

const PATH_CRYPTO_FILES = "secrets/paramsCKKS"

// Suite is the type of keys used to secure communication in Onet
var Suite = edwards25519.NewBlakeSHA256Ed25519()

func TestRegEncryptedTraining(t *testing.T) {
	log.SetDebugVisible(2)

	t.Run("CellCNN", genTest(
		"cellCNN", "../../normalized/",
		3,
		true, false,
		cellCNN.Samples, cellCNN.Cells, cellCNN.Features, cellCNN.Filters, cellCNN.Classes,
		15, 2000,
		true,
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

		if !deterministic{
			rand.Seed(time.Now().Unix())
		}
		params := cellCNN.GenParams()

		cryptoParamsList := cellCNN.ReadOrGenerateCryptoParams(hosts, &params, PATH_CRYPTO_FILES)
		require.NotNil(t, cryptoParamsList)

		for _, s := range servers {
			_, err := s.ProtocolRegister(protoName, func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
				pi, err := decentralized.NewTrainingProtocol(tni)
				if err != nil {
					return nil, err
				}
				protocol := pi.(*decentralized.TrainingProtocol)

				vars := decentralized.InitCellCNNVars{
					Path:           path,
					PartyDataSize:  partyDataSize/tni.Tree().Size(),
					TrainEncrypted: trainEncrypted,
					Epochs:         epoch * cellCNN.Samples / cellCNN.BatchSize,
					Samples:        samples,
					Cells:          cells,
					Features:       features,
					Filters:        filters,
					Classes:        classes,
					Debug:          debug,
				}
				protocol.InitVars(cryptoParamsList[tni.Index()], vars)

				// 1) Load Data
				protocol.XTrain, protocol.YTrain, _, _ = protocol.LoadData()

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
