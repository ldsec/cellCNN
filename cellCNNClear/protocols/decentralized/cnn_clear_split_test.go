package decentralized

import (
	//	"fmt"
	"github.com/ldsec/cellCNN/cellCNNClear/protocols/common"
	"github.com/stretchr/testify/require"
	"go.dedis.ch/kyber/v3/group/edwards25519"
	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"
	//	"os"
	"testing"
)

var Suite = edwards25519.NewBlakeSHA256Ed25519()

func TestCnnSplit(t *testing.T) {
	log.SetDebugVisible(2)

	//common.TrainData = common.LoadCellCnnTrainData()

	// Suite is the type of keys used to secure communication in Onet

	local := onet.NewLocalTest(Suite)
	servers, _, tree := local.GenTree(HOSTS, true)
	defer local.CloseAll()

	CnnClearTestRegister := func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
		return NewCnnSplitTest(tni, NBR_LOCAL_ITER, NBR_EPOCHS, DEBUG)
	}

	for _, s := range servers {
		_, err := s.ProtocolRegister("CnnClearTest", CnnClearTestRegister)
		require.NoError(t, err)
	}

	loader, err := common.GetValidLoader()
	err, _ = RunCnnClearTest(local, nil, tree, TIME, false, "CnnClearTest", 1, loader)
	require.NoError(t, err)
}

/*
func TestCnnSplitEval(t *testing.T) {
	if !eval {
		return
	}

	local_iter_eval := [3]int{1, 5, 10}
	epochs_eval := [6]int{50, 100, 150, 200, 250, 300}
	var f *os.File
	var err error
	f, err = os.Create("split_accuracy_eval.csv")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	f.WriteString("hosts,n_local_iter,batch_size,epochs,kfolds,accuracy,precision,recall,fscore\n")

	//common.TrainData = common.LoadCellCnnTrainData()

	for _, nLocalIter := range local_iter_eval {
		for _, nEpochs := range epochs_eval {
			local := onet.NewLocalTest(Suite)

			servers, _, tree := local.GenTree(HOSTS, true)
			defer local.CloseAll()

			CnnClearTestRegister := func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
				return NewCnnSplitTest(tni, NBR_LOCAL_ITER, NBR_EPOCHS, DEBUG)
			}

			for _, s := range servers {
				_, err := s.ProtocolRegister("CnnClearTest", CnnClearTestRegister)
				require.NoError(t, err)
			}

			loader, _ := common.GetValidLoader()
			_, s_out := RunCnnClearTest(local, nil, tree, true, TIME, "CnnClearTest", 1, loader)
			s_params := fmt.Sprintf("%d,%d,%d,%d,%d,", HOSTS, nLocalIter, common.BATCH_SIZE, nEpochs, 1)
			f.WriteString(s_params + s_out)
			fmt.Println(s_params + s_out)
		}
	}

	require.NoError(t, err)
}
*/
