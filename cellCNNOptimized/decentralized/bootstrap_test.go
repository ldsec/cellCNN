package decentralized_test

import (
	"github.com/ldsec/cellCNN/cellCNNOptimized"
	"github.com/ldsec/cellCNN/cellCNNOptimized/decentralized"
	"go.dedis.ch/onet/v3/log"
	"testing"

	"go.dedis.ch/kyber/v3/suites"

	"github.com/stretchr/testify/require"

	"github.com/ldsec/lattigo/v2/ckks"
	"go.dedis.ch/onet/v3"
)

func TestBootstrapProtocol(t *testing.T) {
	log.SetDebugVisible(2)

	hosts := 3

	params := cellCNN.GenParams()

	cryptoParamsList := cellCNN.NewCryptoParamsForNetwork(&params, hosts)

	local := onet.NewLocalTest(suites.MustFind("Ed25519"))
	defer local.CloseAll()

	servers, _, tree := local.GenTree(hosts, true)
	for sIdx, s := range servers {
		_, err := s.ProtocolRegister("BootstrapTest", decentralized.NewBootstrapProtocolFunction(cryptoParamsList[sIdx]))
		require.NoError(t, err)
	}

	rootInstance, err := local.CreateProtocol("BootstrapTest", tree)
	require.NoError(t, err)

	protocol := rootInstance.(*decentralized.BootstrapProtocol)
	err = protocol.InitRoot(ckks.NewCiphertext(*cryptoParamsList[0].Params, 1, cryptoParamsList[0].Params.MaxLevel(), cryptoParamsList[0].Params.Scale()))
	require.NoError(t, err)

	feedback := protocol.FeedbackChannel
	require.NoError(t, protocol.Start())

	<-feedback
}