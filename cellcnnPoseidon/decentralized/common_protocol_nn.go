package decentralized

import (
	"fmt"

	"go.dedis.ch/onet/v3"
)

// RunNeuralNetworkTest runs a protocol neural network test or simulation
func RunNeuralNetworkTest(
	localTest *onet.LocalTest, overlay *onet.Overlay, tree *onet.Tree, local, time bool, name string,
) ([][]byte, error) {

	nbrRuns := 1
	// if we wish to record the time we only need to run the protocol once
	if time {
		nbrRuns = 1
	}

	wEncrypted := make([][]byte, 0)

	for i := 0; i < nbrRuns; i++ {
		var err error
		var rootInstance onet.ProtocolInstance

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
	}

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
