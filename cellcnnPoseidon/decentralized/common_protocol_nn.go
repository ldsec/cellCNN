package decentralized

import (
	"fmt"

	"go.dedis.ch/onet/v3"
)

// ReadOrGenerateCryptoParamsForNN reads (from a set of files) or generates new cryptoParams for a neural network learning to be given to each node
// func ReadOrGenerateCryptoParamsForNN(hosts int, defaultN *ckks.Parameters, rootPath string, sizeInputLayer, sizeOutputLayer int, sizeHiddenLayer []int, generateRotKeys bool) []*libspindle.CryptoParams {
// 	return spindleprotcommon.ReadOrGenerateCryptoParams(hosts, defaultN, rootPath, protocols.NeuralNetwork, nil, &spindleprotcommon.NNParams{
// 		SizeInputLayer:  sizeInputLayer,
// 		SizeHiddenLayer: sizeHiddenLayer,
// 		SizeOutputLayer: sizeOutputLayer,
// 		Slots:           int(defaultN.Slots()),
// 		ColsPerCipher:   5, //TODO: make this generic
// 	}, generateRotKeys)
// }

// RunNeuralNetworkTest runs a protocol neural network test or simulation
func RunNeuralNetworkTest(
	localTest *onet.LocalTest, overlay *onet.Overlay, tree *onet.Tree, local, time bool, name string,
) ([][]byte, error) {

	// var accuracy, precision, recall, fscore, auc float64
	// nbrRuns := int(kFold)
	nbrRuns := 1
	// if we wish to record the time we only need to run the protocol once
	if time {
		nbrRuns = 1
	}

	wEncrypted := make([][]byte, 0)

	for i := 0; i < nbrRuns; i++ {
		var err error
		var rootInstance onet.ProtocolInstance

		// w := make([][][]float64, 0)
		// running a localTest test
		if localTest != nil {
			rootInstance, err = localTest.CreateProtocol(name, tree)
			if err != nil {
				return nil, err
			}
			// running a simulation
		} else if overlay != nil {
			rootInstance, err = overlay.CreateProtocol(name, tree, onet.NilServiceID)
			if err != nil {
				return nil, err
			}
		}

		// get the final weights after running the protocol
		wEncrypted = runNNEnc(rootInstance)
		fmt.Println("get the final encrypted weights")

		// // if using real data
		// if loader != nil {
		// 	if encrypted {
		// 		protocol := rootInstance.(*NNEncryptedProtocol)
		// 		if version == "v1" {
		// 			w = protocol.DecryptNNFinalWeights(wEncrypted)
		// 		} else if version == "v2" {
		// 			w = protocol.DecryptModel2Layer(wEncrypted)
		// 		}
		// 	}

		// 	accuracyTmp, precisionTmp, recallTmp, fscoreTmp, aucTmp := runNNPredictionTest(spindleprotcommon.TestData.X, spindleprotcommon.TestData.Y, w, classes, sizeLayer, print)
		// 	log.LLvl1(accuracyTmp, precisionTmp, recallTmp, fscoreTmp, aucTmp)
		// 	accuracy += accuracyTmp
		// 	precision += precisionTmp
		// 	recall += recallTmp
		// 	fscore += fscoreTmp
		// 	auc += aucTmp
		// }
	}

	// log.Lvl2("accuracy: ", accuracy/float64(nbrRuns))
	// log.Lvl2("precision:", precision/float64(nbrRuns))
	// log.Lvl2("recall:   ", recall/float64(nbrRuns))
	// log.Lvl2("F-score:  ", fscore/float64(nbrRuns))
	// log.Lvl2("AUC:      ", auc/float64(nbrRuns))

	// stats := make([]float64, 5)
	// stats[0] = accuracy / float64(nbrRuns)
	// stats[1] = precision / float64(nbrRuns)
	// stats[2] = recall / float64(nbrRuns)
	// stats[3] = fscore / float64(nbrRuns)
	// stats[4] = auc / float64(nbrRuns)

	return wEncrypted, nil
}

func runNNEnc(rootInstance onet.ProtocolInstance) [][]byte {
	protocol := rootInstance.(*NNEncryptedProtocol)
	feedback := protocol.FeedbackChannel
	go protocol.Start()
	return <-feedback
}

// runRegPredictionTest evaluates a neural network model with the selected test data
// func runNNPredictionTest(testDataX [][]float64, testDataY []float64, weights [][][]float64, classes, sizeLayer []int, print bool) (float64, float64, float64, float64, float64) {
// 	return libspindle.EvaluateNNModel(testDataX, testDataY, weights, classes, sizeLayer, print)
// }
