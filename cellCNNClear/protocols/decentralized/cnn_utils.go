package decentralized

import (
	"errors"
	"fmt"
	"github.com/ldsec/cellCNN/cellCNNClear/protocols/common"
	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"
)

func NewCNNClearTest(tni *onet.TreeNodeInstance, trainData common.CnnDataset, nbrLocalIter, nbrEpochs int, debug Debug) (onet.ProtocolInstance, error) {
	pi, err := NewCnnClearProtocol(tni)
	if err != nil {
		return nil, err
	}
	protocol := pi.(*CnnClearProtocol)

	// ##STEP 1: Split data
	protocol.X, protocol.Y, protocol.MaxIterations = common.SplitData(trainData.X, trainData.Y, len(tni.Roster().List),
		tni.Index(), nbrEpochs, nbrLocalIter, common.BATCH_SIZE, tni.IsRoot())

	// ##STEP 2: InitRoot protocol training variables
	InitCnnClearProtocolVars(protocol, common.LEARN_RATE, common.MOMENTUM, nbrLocalIter, common.BATCH_SIZE)

	protocol.Debug = debug
	protocol.Debug.ServerID = tni.ServerIdentity().String()
	return protocol, nil
}

func NewCnnSplitTest(tni *onet.TreeNodeInstance, nbrLocalIter, nbrEpochs int, debug Debug) (onet.ProtocolInstance, error) {

	pi, err := NewCnnClearProtocol(tni)
	if err != nil {
		return nil, err
	}
	protocol := pi.(*CnnClearProtocol)

	// ##STEP 1: Split data
	protocol.X, protocol.Y, protocol.MaxIterations = common.LoadSplitData(len(tni.Roster().List),
		tni.Index(), nbrEpochs, nbrLocalIter, common.BATCH_SIZE, tni.IsRoot())

	// ##STEP 2: InitRoot protocol training variables
	InitCnnClearProtocolVars(protocol, common.LEARN_RATE, common.MOMENTUM, nbrLocalIter, common.BATCH_SIZE)

	protocol.Debug = debug
	protocol.Debug.ServerID = tni.ServerIdentity().String()
	return protocol, nil
}

// runCnnClear runs a CNN protocol from the root
func runCnnClear(rootInstance onet.ProtocolInstance, weights common.WeightsVector) common.WeightsVector {
	protocol := rootInstance.(*CnnClearProtocol)

	feedback := protocol.FeedbackChannel
	go protocol.Start()

	wTmp := <-feedback
	weights = wTmp
	return weights
}

// InitCnnClearProtocolVars sets the training parameters for the cleartext version of the regression protocol. We assume data is pre-loaded.
func InitCnnClearProtocolVars(p *CnnClearProtocol, learnRate, momentum float64, nbrLocalIter, batchSize int) error {

	if p.X == nil || len(p.X) <= 0 {
		return errors.New("no data loaded")
	}

	p.LearningRate = learnRate

	p.Conv, p.Pool, p.Dense = common.InitCellCnn(common.NCLASSES)
	p.InitWeights()

	p.LocalIterationNumber = nbrLocalIter
	p.BatchSize = batchSize
	p.momentum = momentum
	p.FirstMoment = false

	return nil
}

// RunCnnClearTest runs a protocol CNN test or simulation
func RunCnnClearTest(localTest *onet.LocalTest, overlay *onet.Overlay, tree *onet.Tree, local, timing bool, name string,
	kFold uint, loader common.Loader) (error, string) {

	var accuracy, precision, recall, fscore, accuracyMulti, precisionMulti, recallMulti, fscoreMulti float64
	nbrRuns := kFold

	// if we wish to record the time we only need to run the protocol once
	if timing {
		nbrRuns = 1
	}

	for i := uint(0); i < nbrRuns; i++ {
		// if ran locally in one machine we only need to load the data once (the struct is shared among nodes)
		//if local && loader != nil {
		//	dataset, err := loader.Load()
		//	dataset.Shuffle(1)
		//	common.TrainData, common.TestData, err = dataset.Partition(kFold, i)
		//	if err != nil {
		//		return fmt.Errorf("when partitioning data: %w", err), ""
		//	}
		//}
		var w common.WeightsVector
		var err error
		var rootInstance onet.ProtocolInstance

		// running a localTest test
		if localTest != nil {
			//fmt.Println("running local test")
			rootInstance, err = localTest.CreateProtocol(name, tree)
			if err != nil {
				return err, ""
			}
		} else if overlay != nil {
			rootInstance, err = overlay.CreateProtocol(name, tree, onet.NilServiceID)
			if err != nil {
				return err, "error while registering protocol instance to overlay"
			}
		}

		w = runCnnClear(rootInstance, w)

		//accuracyTmp, precisionTmp, recallTmp, fscoreTmp := common.RunCnnClearPredictionTest(w, common.TestData.X, common.TestData.Y)

		testAllData := common.LoadCellCnnTestAll(common.DATA_FOLDER, common.TESTALLCELL, common.NFEATURES, common.TESTSAMPLES)
		testMultiData := common.LoadCellCnnValidData(common.DATA_FOLDER, common.NSAMPLES, common.NCELLS, common.NFEATURES, common.TYPEDATA)
		accuracyTmpMulti, precisionTmpMulti, recallTmpMulti, fscoreTmpMulti := common.RunCnnClearPredictionTestAll(w, testMultiData, common.NCLASSES)
		log.Lvlf2("Multi-cell test data results:")
		log.LLvl1(accuracyTmpMulti, precisionTmpMulti, recallTmpMulti, fscoreTmpMulti)

		log.Lvlf2("All test data results:")
		accuracyTmp, precisionTmp, recallTmp, fscoreTmp := common.RunCnnClearPredictionTestAll(w, testAllData, common.NCLASSES)
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
