package decentralized

import (
	"fmt"
	"github.com/ldsec/cellCNN/cellCNNClear/protocols/common"
	"github.com/stretchr/testify/require"
	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"
	"os"
	"testing"
)

var DEBUG = Debug{Print: true}

func TestEvalCNN(t *testing.T) {
	if !eval {
		return
	}

	hosts_eval := [3]int{3}
	local_iter_eval := [3]int{1}
	epochs_eval := [5]int{10}
	var f *os.File
	var err error
	f, err = os.Create("accuracy_eval.csv")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	f.WriteString("hosts,n_local_iter,batch_size,epochs,kfolds,accuracy,precision,recall,fscore\n")

	common.TrainData = common.LoadCellCnnTrainData()

	for _, nHosts := range hosts_eval {
		for _, nLocalIter := range local_iter_eval {
			for _, nEpochs := range epochs_eval {
				local := onet.NewLocalTest(Suite)

				servers, _, tree := local.GenTree(nHosts, true)
				defer local.CloseAll()

				CnnClearTestRegister := func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
					return NewCNNClearTest(tni, common.TrainData, nLocalIter, nEpochs, DEBUG)
				}

				for _, s := range servers {
					_, err := s.ProtocolRegister("CnnClearTest", CnnClearTestRegister)
					require.NoError(t, err)
				}

				loader, _ := common.GetLoader()
				_, s_out := RunCnnClearTest(local, nil, tree, true, TIME, "CnnClearTest", KFOLDS, loader)
				s_params := fmt.Sprintf("%d,%d,%d,%d,%d,", nHosts, nLocalIter, common.BATCH_SIZE, nEpochs, KFOLDS)
				f.WriteString(s_params + s_out)
				fmt.Println(s_params + s_out)
			}
		}
	}

	require.NoError(t, err)
}

func TestCNNClear(t *testing.T) {
	log.SetDebugVisible(2)

	common.TrainData = common.LoadCellCnnTrainData()

	local := onet.NewLocalTest(Suite)
	servers, _, tree := local.GenTree(HOSTS, true)
	defer local.CloseAll()

	CnnClearTestRegister := func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
		return NewCNNClearTest(tni, common.TrainData, NBR_LOCAL_ITER, NBR_EPOCHS, DEBUG)
	}

	for _, s := range servers {
		_, err := s.ProtocolRegister("CnnClearTest", CnnClearTestRegister)
		require.NoError(t, err)
	}

	loader, err := common.GetLoader()
	err, _ = RunCnnClearTest(local, nil, tree, TIME, true, "CnnClearTest", KFOLDS, loader)
	require.NoError(t, err)
}
