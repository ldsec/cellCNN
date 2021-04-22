package centralized

import (
	"fmt"
	"github.com/ldsec/cellCNN/cellCNN_clear/layers"
	"github.com/ldsec/cellCNN/cellCNN_clear/protocols/common"
	"github.com/ldsec/cellCNN/cellCNN_clear/utils"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"os"
	"time"
)

const learn_rate = 0.1
const batchSize = 200
const write = false // write accuracy values to file

func ComputeGradient(i int, j int, v float64, y []float64) float64 {
	//represent one hot encoding

	if int(y[i]) == 0 {
		if int(y[i]) == j {
			return v - 1
		} else {
			return v
		}
	}
	if int(y[i]) == 1 {
		if int(y[i]) != j {
			return v
		} else {
			return v - 1
		}
	}
	return 0
}

// Train a cellCNN on given dataset
func Train(dataset common.CnnDataset, validData common.CnnDataset, nclasses int, niter int, learn_rate float64, timing bool) (*mat.Dense, *mat.Dense) {

	X := dataset.X
	y := dataset.Y

	conv := layers.Conv1D{Nfilters: common.NFILTERS}
	pool := layers.Pool{}
	dense := layers.Dense_n{Nclasses: nclasses, ApproxInterval: common.ApproxInterval}

	var delta1 []*mat.Dense
	var delta2 *mat.Dense
	var out1 []*mat.Dense
	var out2 *mat.Dense
	var f *os.File
	var err error
	if write {
		dir := "eval"
		os.Mkdir(dir, 0777)
		f, err = os.Create(dir + "/centralized_eval")
		if err != nil {
			panic(err)
		}
		f.WriteString("iteration,train_acc,test_acc\n")
		defer f.Close()
	}

	for i := 1; i <= niter; i++ {

		// make a new batch
		newBatch := make([]*mat.Dense, batchSize)
		newBatchLabels := make([]float64, batchSize)
		for j := 0; j < len(newBatch); j++ {
			randi := rand.Intn(len(X))
			newBatch[j] = X[randi]
			newBatchLabels[j] = y[randi]
		}

		// forward pass
		out1 = conv.Forward(newBatch, nil)
		out2 = pool.Forward(out1)
		out2 = dense.Forward(out2, nil)
		//fmt.Println("out is" , out2)
		//	fmt.Println("labels are,",newBatchLabels)
		//compute loss gradient + print accuracy
		compute_grad := func(i int, j int, v float64) float64 {
			return ComputeGradient(i, j, v, newBatchLabels)
		}

		var gradient mat.Dense
		gradient.Apply(compute_grad, out2)
		//fmt.Println("gradient is", gradient)
		//	os.Exit(0)
		if i == 1 || i%100 == 0 {
			if !timing {
				fmt.Printf("Iteration: %d \n", i)
				utils.Print_train_stats_cellCNN(out2, newBatchLabels)
			}
		}

		// write training and validation accuracy to file
		if write {
			newValidBatch := make([]*mat.Dense, batchSize)
			newValidBatchLabels := make([]float64, batchSize)
			for j := 0; j < len(newBatch); j++ {
				randi := rand.Intn(len(validData.X))
				newValidBatch[j] = validData.X[randi]
				newValidBatchLabels[j] = validData.Y[randi]
			}

			weights := common.WeightsVector{conv.GetWeights(), dense.GetWeights()}
			trainAcc, _, _, _ := common.RunCnnClearPredictionTest(weights, newBatch, newBatchLabels)
			validAcc, _, _, _ := common.RunCnnClearPredictionTest(weights, newValidBatch, newValidBatchLabels)
			s := fmt.Sprintf("%d, %.2f, %.2f\n", i, trainAcc, validAcc)
			f.WriteString(s)
		}

		// backpropagation
		delta2, _ = dense.Backward(&gradient, learn_rate, common.MOMENTUM)
		delta1 = pool.Backward(delta2)
		conv.Backward(delta1, learn_rate, common.MOMENTUM)
	}
	//common.PrintM(conv.GetWeights())
	return conv.GetWeights(), dense.GetWeights()

}

// input: nsamples x ncells x nmarkers
// conv1D + dense
func cellCNN(nepochs int, timing bool) {
	niter := nepochs * common.NSAMPLES / batchSize
	fmt.Printf("%d epochs -> %d iterations (with batch size %d)\n", nepochs, niter, batchSize)

	startLoad := time.Now()
	trainData := common.LoadCellCnnTrainData()
	validData := common.LoadCellCnnValidData()
	timeLoad := time.Since(startLoad)

	startTrain := time.Now()
	weights := make(common.WeightsVector, 2)
	weights[0], weights[1] = Train(trainData, validData, 2, niter, learn_rate, timing)
	timeTrain := time.Since(startTrain)

	if timing {
		fmt.Printf("Loading the data: %s\nTraining: %s (%d iterations)\n", timeLoad, timeTrain, niter)
	}

	if !timing {
		testSize := 1000
		//fmt.Print(len(validData.X))
		newBatch := make([]*mat.Dense, testSize)
		newBatchLabels := make([]float64, testSize)
		for j := 0; j < len(newBatch); j++ {
			randi := rand.Intn(len(validData.X))
			newBatch[j] = validData.X[randi]
			newBatchLabels[j] = validData.Y[randi]
		}
		fmt.Println("new batch testing")
		accuracy, precision, recall, fscore := common.RunCnnClearPredictionTest(weights, newBatch, newBatchLabels)
		fmt.Printf("\nValidation\naccuracy: %.2f, precision: %.2f, recall: %.2f, fscore: %.2f\n", accuracy, precision, recall, fscore)
	}
}
