package centralized

import (
	"fmt"
	"github.com/ldsec/cellCNN/cellCNNClear/layers"
	"github.com/ldsec/cellCNN/cellCNNClear/protocols/common"
	"github.com/ldsec/cellCNN/cellCNNClear/utils"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"os"
	"time"
)

const learn_rate = common.LEARN_RATE
const batchSize = common.BATCH_SIZE
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
	if int(y[i]) == 2 {
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
			//fmt.Println(y[randi])
		}

		// forward pass
		out1 = conv.Forward(newBatch, nil)
		out2 = pool.Forward(out1)
		out2 = dense.Forward(out2, nil)

		compute_grad := func(i int, j int, v float64) float64 {
			return ComputeGradient(i, j, v, newBatchLabels)
		}
		//for lab:=0; lab<len(newBatchLabels);lab++{
		//	if newBatchLabels[lab] == 0{
		//		changeIn := out2.At(lab,0)
		//	}
		//	if newBatchLabels[lab] == 1{
		//		changeIn := out2.At(lab,1)
		//	}
		//	if newBatchLabels[lab] == 2{
		//		changeIn := out2.At(lab,2)
		//	}
		//}

		var gradient mat.Dense
		gradient.Apply(compute_grad, out2)
		//fmt.Println("newBatchLabels:")
		//fmt.Println(newBatchLabels[0])
		//fmt.Println(newBatchLabels[1])
		//fmt.Println(newBatchLabels[2])
		//m:=gradient
		//_, c := m.Dims()
		//fmt.Println("Gradients:")
		//for i := 0; i < 3; i++ {
		//	for j := 0; j < c; j++ {
		//		print(m.At(i, j), "\t")
		//	}
		//	println("")
		//}
		//println("")
		//
		//_, c = out2.Dims()
		//fmt.Println("Outs:")
		//for i := 0; i < 3; i++ {
		//	for j := 0; j < c; j++ {
		//		print(out2.At(i, j), "\t")
		//	}
		//	println("")
		//}
		if i == 1 || i%batchSize == 0 {
			if !timing {
				fmt.Printf("Iteration: %d \n", i)
				utils.Print_train_stats_cellCNN(out2, newBatchLabels, common.NCLASSES, common.MICRO)
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
	testAllData := common.LoadCellCnnTestAll()

	fmt.Println("loaded")
	//fmt.Println(reflect.TypeOf(A))
	//fmt.Println(A[0])

	timeLoad := time.Since(startLoad)

	startTrain := time.Now()
	weights := make(common.WeightsVector, 2)
	weights[0], weights[1] = Train(trainData, validData, common.NCLASSES, niter, learn_rate, timing)
	timeTrain := time.Since(startTrain)

	if timing {
		fmt.Printf("Loading the data: %s\nTraining: %s (%d iterations)\n", timeLoad, timeTrain, niter)
	}

	if !timing {
		testSize := 2250
		//fmt.Print(len(validData.X))
		newBatch := make([]*mat.Dense, testSize)
		newBatchLabels := make([]float64, testSize)
		for j := 0; j < len(newBatch); j++ {
			randi := rand.Intn(len(validData.X))
			newBatch[j] = validData.X[randi]
			newBatchLabels[j] = validData.Y[randi]
		}
		fmt.Println("Test acc. on multi-cell inputs with 200 cells each")
		accuracy, precision, recall, fscore := common.RunCnnClearPredictionTest(weights, newBatch, newBatchLabels)
		fmt.Printf("\nTest\naccuracy: %.2f, precision: %.2f, recall: %.2f, fscore: %.2f\n", accuracy, precision, recall, fscore)
		fmt.Println("Test acc. on 6 patients with >5000 cells each")
		accuracy, precision, recall, fscore = common.RunCnnClearPredictionTestAll(weights, testAllData)
		fmt.Printf("\nTest All\naccuracy: %.2f, precision: %.2f, recall: %.2f, fscore: %.2f\n", accuracy, precision, recall, fscore)
	}
}
