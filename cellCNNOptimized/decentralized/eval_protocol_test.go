package decentralized_test

import (
	"fmt"
	"github.com/ldsec/cellCNN/cellCNNClear/protocols/common"
	"github.com/ldsec/cellCNN/cellCNNOptimized/decentralized"
	"github.com/stretchr/testify/require"
	"go.dedis.ch/kyber/v3/group/edwards25519"
	"go.dedis.ch/onet/v3"
	"os"
	"testing"
)

var Suite = edwards25519.NewBlakeSHA256Ed25519()

const HOSTS = 3
const NBR_LOCAL_ITER = 1
const NBR_EPOCHS = 10
const KFOLDS = 1
const DEBUG = false

func TestEvalCNN(t *testing.T) {
	var f *os.File
	var err error
	f, err = os.Create("accuracy_eval.csv")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	f.WriteString("hosts,n_local_iter,batch_size,epochs,kfolds,accuracy,precision,recall,fscore\n")

	common.TrainData = common.LoadCellCnnTrainData()

	local := onet.NewLocalTest(Suite)

	servers, _, tree := local.GenTree(HOSTS, true)
	defer local.CloseAll()

	CnnClearTestRegister := func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
		return NewCNNEncryptedTest(tni, common.TrainData, NBR_LOCAL_ITER, NBR_EPOCHS, DEBUG)
	}

	for _, s := range servers {
		_, err := s.ProtocolRegister("CnnEncryptedTest", CnnClearTestRegister)
		require.NoError(t, err)
	}

	loader, _ := common.GetLoader()
	_, s_out := RunCnnEncryptedTest(local, nil, tree, true, TIME, "CnnClearTest", KFOLDS, loader)
	s_params := fmt.Sprintf("%d,%d,%d,%d,%d,", HOSTS, NBR_LOCAL_ITER, common.BATCH_SIZE, NBR_EPOCHS, KFOLDS)
	f.WriteString(s_params + s_out)
	fmt.Println(s_params + s_out)

	require.NoError(t, err)
}

func NewCNNEncryptedTest(tni *onet.TreeNodeInstance, trainData common.CnnDataset, nbrLocalIter, nbrEpochs int, debug bool) (onet.ProtocolInstance, error) {
	pi, err := decentralized.NewTrainingProtocol(tni)
	if err != nil {
		return nil, err
	}
	protocol := pi.(*decentralized.TrainingProtocol)

	// ##STEP 1: Split data
	protocol.X, protocol.Y, protocol.MaxIterations = common.SplitData(trainData.X, trainData.Y, len(tni.Roster().List),
		tni.Index(), nbrEpochs, nbrLocalIter, common.BATCH_SIZE, tni.IsRoot())

	// ##STEP 2: InitRoot protocol training variables
	InitCnnClearProtocolVars(protocol, common.LEARN_RATE, common.MOMENTUM, nbrLocalIter, common.BATCH_SIZE)

	protocol.Debug = debug
	return protocol, nil
}
