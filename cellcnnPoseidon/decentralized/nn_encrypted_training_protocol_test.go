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

func TestDemo(t *testing.T) {
	protoID := "BCW"
	protoName := "NNEncryptedTest/" + protoID

	local := onet.NewLocalTest(GetSuit())
	defer local.CloseAll()

	var err error

	cryptoList := CustomizedNetworkKeysList(HOSTS)

	servers, _, tree := local.GenTree(HOSTS, true)
	for _, s := range servers {
		_, err := s.ProtocolRegister(protoName, func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
			pi, err := NewNNEncryptedProtocol(tni)
			if err != nil {
				return nil, err
			}
			protocol := pi.(*NNEncryptedProtocol)

			// using randomly generated data for training
			protocol.TrainSet = nil
			protocol.Sync = true

			// 2. init protocol training variables
			err = protocol.Init("v1", cryptoList[tni.Index()])
			return protocol, err
		})
		require.NoError(t, err)

		// _, err = s.ProtocolRegister(decentralized.BootstrapProtocolName, decentralized.NewBootstrapProtocolFunction(cryptoParamsList[sIdx], seed))
		// require.NoError(t, err)
	}

	fmt.Println("start running")
	_, err = RunNeuralNetworkTest(local, nil, tree, true, true, protoName)

	require.NoError(t, err)
}
