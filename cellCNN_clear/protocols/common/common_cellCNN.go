package common

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/ldsec/cellCNN/cellCNN_clear/layers"
	"github.com/ldsec/cellCNN/cellCNN_clear/utils"
	"go.dedis.ch/onet/v3/log"
	"gonum.org/v1/gonum/mat"
)

type WeightsVector []*mat.Dense

type CnnDataset struct {
	X []*mat.Dense
	Y []float64
}

var TrainData CnnDataset
var TestData CnnDataset

// Partition split dataset's row into n groups, return (trainData, testData)
func (dataset *CnnDataset) Partition(numberOfGroup, testGroup uint) (CnnDataset, CnnDataset, error) {
	if numberOfGroup == 0 {
		return CnnDataset{}, CnnDataset{}, errors.New("number of group is 0")
	}
	// no kfold:
	if numberOfGroup == 1 {
		return *dataset, *dataset, nil
	}
	if numberOfGroup <= testGroup {
		return CnnDataset{}, CnnDataset{}, errors.New("currentGroup is greater than number of group")
	}

	groupSize := uint(len(dataset.X)) / numberOfGroup
	if groupSize == 0 {
		return CnnDataset{}, CnnDataset{}, errors.New("numberOfGroup is greater than number of DataSet's row")
	}

	startTest, endTest := groupSize*testGroup, groupSize*(testGroup+1)

	testData := CnnDataset{
		X: dataset.X[startTest:endTest],
		Y: dataset.Y[startTest:endTest],
	}

	var trainData CnnDataset
	switch testGroup {
	case 0:
		trainData = CnnDataset{
			X: dataset.X[endTest:],
			Y: dataset.Y[endTest:],
		}
	case numberOfGroup - 1:
		trainData = CnnDataset{
			X: dataset.X[:startTest],
			Y: dataset.Y[:startTest],
		}
	default:
		trainData = CnnDataset{
			X: append(dataset.X[:startTest], dataset.X[endTest:]...),
			Y: append(dataset.Y[:startTest], dataset.Y[endTest:]...),
		}
	}

	return trainData, testData, nil
}

// Shuffle shuffles the row inplace, depending on the given seed
func (dataset *CnnDataset) Shuffle(seed int64) {
	rand.Seed(seed)
	rand.Shuffle(len(dataset.X), func(i, j int) {
		dataset.X[i], dataset.X[j] = dataset.X[j], dataset.X[i]
		dataset.Y[i], dataset.Y[j] = dataset.Y[j], dataset.Y[i]
	})
}

func LoadCellCnnTrainData() CnnDataset {
	y_tr := utils.String_to_float(utils.Load_file(DATA_FOLDER+"y_train.txt", NSAMPLES))
	X_tr := make([]*mat.Dense, NSAMPLES)
	var fname string
	for i := range X_tr {
		fname = fmt.Sprintf("X_train/%d.txt", i)
		X_tr[i] = utils.Convert_X_cellCNN(utils.Load_file(DATA_FOLDER+fname, NCELLS), NCELLS, NFEATURES, false)
	}
	return CnnDataset{X: X_tr, Y: y_tr}
}

func LoadTrainDataFrom(path string) CnnDataset {
	y_tr := utils.String_to_float(utils.Load_file(path+"y_train.txt", NSAMPLES))
	X_tr := make([]*mat.Dense, NSAMPLES)
	var fname string
	for i := range X_tr {
		fname = fmt.Sprintf("X_train/%d.txt", i)
		X_tr[i] = utils.Convert_X_cellCNN(utils.Load_file(path+fname, NCELLS), NCELLS, NFEATURES, false)
	}
	return CnnDataset{X: X_tr, Y: y_tr}
}

func LoadSplitCellCnnTrainData(hostNumber int) ([]*mat.Dense, []float64) {
	dataFolder := SPLIT_DATA_FOLDER + fmt.Sprintf("host%d/", hostNumber)
	y_tr := utils.String_to_float(utils.Load_file(dataFolder+"y_train.txt", NSAMPLES_DIST))
	X_tr := make([]*mat.Dense, NSAMPLES_DIST)
	var fname string
	for i := range X_tr {
		fname = fmt.Sprintf("X_train/%d.txt", i)
		X_tr[i] = utils.Convert_X_cellCNN(utils.Load_file(dataFolder+fname, NCELLS), NCELLS, NFEATURES, false)
	}
	return X_tr, y_tr
}

func LoadCellCnnValidData() CnnDataset {
	y_tr := utils.String_to_float(utils.Load_file(DATA_FOLDER+"y_valid.txt", NSAMPLES))
	X_tr := make([]*mat.Dense, NSAMPLES)
	var fname string
	for i := range X_tr {
		fname = fmt.Sprintf("X_valid/%d.txt", i)
		X_tr[i] = utils.Convert_X_cellCNN(utils.Load_file(DATA_FOLDER+fname, NCELLS), NCELLS, NFEATURES, false)
	}
	return CnnDataset{X: X_tr, Y: y_tr}
}
func LoadCellCnnTestAll() CnnDataset {
	samples_test := 6
	ncell := testAllCell
	y_tr := utils.String_to_float(utils.Load_file(DATA_FOLDER+"y_test_all.txt", samples_test))
	X_tr := make([]*mat.Dense, samples_test)
	var fname string
	for i := range X_tr {
		fname = fmt.Sprintf("X_test_all/%d.txt", i)
		X_tr[i] = utils.Convert_X_cellCNN(utils.Load_file_Big(DATA_FOLDER+fname, ncell), ncell, NFEATURES, true)
	}
	return CnnDataset{X: X_tr, Y: y_tr}
}

// RunCnnClearPredictionTest returns the accuracy, precision, recall given the weights
func RunCnnClearPredictionTest(w WeightsVector, x []*mat.Dense, y []float64) (float64, float64, float64, float64) {
	conv, pool, dense := InitCellCnn()
	println(x[0])

	out1 := conv.Forward(x, w[0])
	out2 := pool.Forward(out1)
	output := dense.Forward(out2, w[1])

	classified := utils.ClassifyCellCNN(output, NCLASSES)
	accuracy := utils.ComputeAccuracy(classified, y)
	precision, recall := utils.ComputePrecisionRecall(classified, y, NCLASSES, MICRO)
	fscore := 2 * precision * recall / (precision + recall)

	return accuracy, precision, recall, fscore
}

// RunCnnClearPredictionTest returns the accuracy, precision, recall given the weights for all test donors (phenotype prediction)
func RunCnnClearPredictionTestAll(w WeightsVector, dataset CnnDataset) (float64, float64, float64, float64) {
	x := dataset.X
	y := dataset.Y

	conv, pool, dense := InitCellCnn()
	out1 := conv.Forward(x, w[0])
	//fmt.Println(out1)

	out2 := pool.Forward(out1)
	fmt.Printf(" %v\n", mat.Formatted(out2, mat.Prefix(" "), mat.Excerpt(3)))

	output := dense.Forward(out2, w[1])
	fmt.Printf(" %v\n", mat.Formatted(output, mat.Prefix(" "), mat.Excerpt(3)))

	classified := utils.ClassifyCellCNN(output, NCLASSES)
	accuracy := utils.ComputeAccuracy(classified, y)
	precision, recall := utils.ComputePrecisionRecall(classified, y, NCLASSES, MICRO)
	fscore := 2 * precision * recall / (precision + recall)

	return accuracy, precision, recall, fscore
}
func InitCellCnn() (layers.Conv1D, layers.Pool, layers.Dense_n) {
	conv := layers.Conv1D{Nfilters: NFILTERS}
	pool := layers.Pool{}
	dense := layers.Dense_n{Nclasses: NCLASSES, ApproxInterval: ApproxInterval}
	return conv, pool, dense
}

// SplitData splits the dataset in multiple chunks each assigned to a different node and calculates the necessary number
// of global iterations needed to process the data <nbrDatasetUsed> times with a given number of local iterations (<nbrLocalIter>) and batch size (<nodeBatchSize>)

func SplitData(trainX []*mat.Dense, trainY []float64, nbrNodes int, index, nbrDatasetUsed, nbrLocalIter, nodeBatchSize int, isRoot bool) ([]*mat.Dense, []float64, int) {
	dataRecordsAtEachNode := len(trainX) / nbrNodes
	dataNodeIndex := dataRecordsAtEachNode * index

	maxIterations := 0
	maxIterations = int(math.Ceil(float64(nbrDatasetUsed*dataRecordsAtEachNode) / float64(nbrLocalIter*nodeBatchSize)))
	if maxIterations == 0 {
		maxIterations = 1
	}

	if isRoot {

		log.Lvl2("Each node has ", dataRecordsAtEachNode, " records")
		log.Lvl2("To use the entire dataset ", nbrDatasetUsed, " times, the number of protocol iterations will be ", maxIterations)
	}
	log.Lvl2("Node ", index, " has records between ", 0+dataRecordsAtEachNode*index, " to ", dataRecordsAtEachNode+dataRecordsAtEachNode*index)

	var X []*mat.Dense
	var Y []float64
	X = trainX[dataNodeIndex : dataNodeIndex+dataRecordsAtEachNode]
	Y = trainY[dataNodeIndex : dataNodeIndex+dataRecordsAtEachNode]

	return X, Y, maxIterations
}
func PrintM(m *mat.Dense) {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			print(m.At(i, j), "\t")
		}
		println("")
	}
	println("")
}
func printS() {

	println("")
}
func LoadSplitData(nbrNodes int, index, nbrDatasetUsed, nbrLocalIter, nodeBatchSize int, isRoot bool) ([]*mat.Dense, []float64, int) {
	dataRecordsAtEachNode := NSAMPLES_DIST

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

	X, Y := LoadSplitCellCnnTrainData(index)

	return X, Y, maxIterations
}
