package decentralized_test

import (
	"fmt"
	"github.com/ldsec/cellCNN/cellCNNClear/protocols/common"
	"github.com/ldsec/cellCNN/cellCNNOptimized"
	"github.com/ldsec/cellCNN/cellCNNOptimized/decentralized"
	"github.com/stretchr/testify/require"
	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
	"testing"
)

const HOSTS = 3
const NBR_LOCAL_ITER = 1
const NBR_EPOCHS = 10
const DEBUG = false
const TIME = false


func TestCnnSplit(t *testing.T) {
	log.SetDebugVisible(2)

	rand.Seed(0)

	local := onet.NewLocalTest(Suite)
	servers, _, tree := local.GenTree(HOSTS, true)
	defer local.CloseAll()

	params := cellCNN.GenParams()
	cryptoParamsList := cellCNN.ReadOrGenerateCryptoParams(HOSTS, &params, PathCryptoFiles)
	require.NotNil(t, cryptoParamsList)

	// cellCNN parameters
	cellCNN.Cells = 200
	cellCNN.Features = 37
	cellCNN.Samples = 1000
	cellCNN.Classes = 2
	cellCNN.Filters = 8
	cellCNN.BatchSize = 100
	cellCNN.LearningRate = 0.0001
	cellCNN.Momentum = 0.9

	for _, s := range servers {
		_, err := s.ProtocolRegister("CnnEncryptedTest", func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
			pi, err := decentralized.NewTrainingProtocol(tni)
			if err != nil {
				return nil, err
			}
			protocol := pi.(*decentralized.TrainingProtocol)

			// ##STEP 1: Split data
			var maxIterations int
			protocol.XTrain, protocol.YTrain, maxIterations = LoadSplitData(tni.Index(), NBR_EPOCHS, NBR_LOCAL_ITER, common.BATCH_SIZE, protocol.IsRoot())

			// ##STEP 2: InitRoot protocol training variables
			vars := decentralized.InitCellCNNVars{
				TrainPlain:     true,
				TrainEncrypted: false,
				Deterministic:  true,
				MaxIterations:  maxIterations,
				LocalSamples:   cellCNN.Samples / HOSTS,
				Debug:          DEBUG,
			}
			protocol.InitVars(cryptoParamsList[tni.Index()], &params, vars)

			protocol.Debug = DEBUG
			return protocol, nil
		})
		require.NoError(t, err)
	}

	loader, err := common.GetValidLoader()
	err, _ = RunCnnEncTest(local, tree, TIME, "CnnEncryptedTest", 1, loader)
	require.NoError(t, err)
}

func LoadSplitData(index, nbrDatasetUsed, nbrLocalIter, nodeBatchSize int, isRoot bool) ([]*cellCNN.Matrix, []*cellCNN.Matrix, int) {
	dataRecordsAtEachNode := common.NSAMPLES_DIST

	maxIterations := 0
	maxIterations = int(math.Ceil(float64(nbrDatasetUsed*dataRecordsAtEachNode) / float64(nbrLocalIter*nodeBatchSize)))
	if maxIterations == 0 {
		maxIterations = 1
	}

	if isRoot {
		log.LLvl2("nbrDatasetUsed is", nbrDatasetUsed)
		log.Lvl2("Each node has ", dataRecordsAtEachNode, " records")
		log.Lvl2("To use the entire dataset ", nbrDatasetUsed, " times, the number of protocol iterations will be ", maxIterations)
	}

	dataFolder := "../../cellCNNClear/data/cellCNN/splitNK/" + fmt.Sprintf("host%d/", index)
	X, Y := cellCNN.LoadTrainDataFrom(dataFolder, common.NSAMPLES_DIST, cellCNN.Cells, cellCNN.Features)

	return X, Y, maxIterations
}

func convertMatrixToDense(matrix *cellCNN.Matrix) *mat.Dense{
	data := make([]float64, len(matrix.M))
	for i := range matrix.M {
		data[i] = real(matrix.M[i])
	}
	return mat.NewDense(matrix.Rows, matrix.Cols, data)
}

// runCnnEnc runs a CNN protocol from the root
func runCnnEnc(rootInstance onet.ProtocolInstance, weights common.WeightsVector) common.WeightsVector {
	protocol := rootInstance.(*decentralized.TrainingProtocol)

	feedback := protocol.FeedbackChannel
	go protocol.Start()

	w := <-feedback

	return common.WeightsVector{convertMatrixToDense(w[0]), convertMatrixToDense(w[1])}
}

// RunCnnEncTest runs a protocol CNN encrypted test or simulation
func RunCnnEncTest(localTest *onet.LocalTest, tree *onet.Tree, timing bool, name string,
	kFold uint, loader common.Loader) (error, string) {

	var accuracy, precision, recall, fscore, accuracyMulti, precisionMulti, recallMulti, fscoreMulti float64
	nbrRuns := kFold

	// if we wish to record the time we only need to run the protocol once
	if timing {
		nbrRuns = 1
	}

	for i := uint(0); i < nbrRuns; i++ {
		var w common.WeightsVector
		var err error
		var rootInstance onet.ProtocolInstance

		//fmt.Println("running local test")
		rootInstance, err = localTest.CreateProtocol(name, tree)
		if err != nil {
			return err, ""
		}

		w = runCnnEnc(rootInstance, w)
		//accuracyTmp, precisionTmp, recallTmp, fscoreTmp := common.RunCnnClearPredictionTest(w, common.TestData.X, common.TestData.Y)

		testAllData := common.LoadCellCnnTestAll()
		testMultiData := common.LoadCellCnnValidData()
		accuracyTmpMulti, precisionTmpMulti, recallTmpMulti, fscoreTmpMulti := common.RunCnnClearPredictionTestAll(w, testMultiData)
		log.Lvlf2("Multi-cell test data results:")
		log.LLvl1(accuracyTmpMulti, precisionTmpMulti, recallTmpMulti, fscoreTmpMulti)

		log.Lvlf2("All test data results:")
		accuracyTmp, precisionTmp, recallTmp, fscoreTmp := common.RunCnnClearPredictionTestAll(w, testAllData)
		log.LLvl1(accuracyTmp, precisionTmp, recallTmp, fscoreTmp)
		accuracy += accuracyTmp
		precision += precisionTmp
		recall += recallTmp
		fscore += fscoreTmp

		accuracyMulti += accuracyTmpMulti
		precisionMulti += precisionTmpMulti
		recallMulti += recallTmpMulti
		fscoreMulti += fscoreTmpMulti

	}

	accuracy = accuracy / float64(nbrRuns)
	precision = precision / float64(nbrRuns)
	recall = recall / float64(nbrRuns)
	fscore = fscore / float64(nbrRuns)

	accuracyMulti = accuracyMulti / float64(nbrRuns)
	precisionMulti = precisionMulti / float64(nbrRuns)
	recallMulti = recallMulti / float64(nbrRuns)
	fscoreMulti = fscoreMulti / float64(nbrRuns)
	log.Lvlf2("All test data results:")
	log.Lvlf2("accuracy: %.2f", accuracy)
	log.Lvlf2("precision: %.2f", precision)
	log.Lvlf2("recall: %.2f", recall)
	log.Lvlf2("fscore: %.2f", fscore)

	log.Lvlf2("Multi-cell test data results:")
	s := fmt.Sprintf("%.2f,%.2f,%.2f,%.2f\n", accuracy, precision, recall, fscore)
	log.Lvlf2("accuracy: %.2f", accuracyMulti)
	log.Lvlf2("precision: %.2f", precisionMulti)
	log.Lvlf2("recall: %.2f", recallMulti)
	log.Lvlf2("fscore: %.2f", fscoreMulti)

	s = fmt.Sprintf("%.2f,%.2f,%.2f,%.2f\n", accuracy, precision, recall, fscore)
	return nil, s
}
