package decentralized_test

import (
	"fmt"
	"github.com/ldsec/cellCNN/cellCNN_optimized/decentralized"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/stretchr/testify/require"
	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"
	"testing"

	"go.dedis.ch/kyber/v3/group/edwards25519"
)

// Suite is the type of keys used to secure communication in Onet
var Suite = edwards25519.NewBlakeSHA256Ed25519()

func genTest(
	protoID string,
	hosts int,
	trainEncrypted, deterministic bool,
	debug bool,
) func(*testing.T) {
	protoName := "CellCNN_optimized/" + protoID

	return func(t *testing.T) {
		local := onet.NewLocalTest(Suite)
		defer local.CloseAll()

		servers, _, tree := local.GenTree(hosts, true)

		cryptoParamsList := regenc.ReadOrGenerateCryptoParamsForReg(HOSTS, N, ALGORITHM_VERSION, PATH_CRYPTO_FILES, trainType, nodeBatchSize, d, true, true)
		require.NotNil(t, cryptoParamsList)

		for _, s := range servers {
			_, err := s.ProtocolRegister(protoName, func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
				pi, err := decentralized.NewTrainingProtocol(tni)
				if err != nil {
					return nil, err
				}
				protocol := pi.(*decentralized.TrainingProtocol)


				protocol.Debug = debug
				return protocol, err
			})
			require.NoError(t, err)
		}

		protocol, err := local.CreateProtocol(protoName, tree)
		require.NoError(t, err)

		feedback := protocol.FeedbackChannel
		if err := protocol.Start(); err != nil {
			log.Panic(fmt.Errorf("start protocol: %v", err))
		}
		<-feedback

	}
}
