package common

import cellCNN "github.com/ldsec/cellCNN/cellCNNOptimized"

// cellCNN parameters
var NCELLS = cellCNN.Cells
var NFEATURES = cellCNN.Features
var NSAMPLES = cellCNN.Samples
var NSAMPLES_DIST = cellCNN.NSamplesDist
var TESTSAMPLES = cellCNN.TestSamples
var NCLASSES = cellCNN.Classes
var TYPEDATA = cellCNN.TypeData
var NFILTERS = cellCNN.Filters
var DATA_FOLDER = cellCNN.DataFolder
var SPLIT_DATA_FOLDER = cellCNN.SplitDataFolder

const ApproxInterval = 3. //do not change this without changing coefficients!
var TESTALLCELL = cellCNN.TestAllCells
var BATCH_SIZE = cellCNN.BatchSize
var LEARN_RATE = cellCNN.LearningRate
var MOMENTUM = cellCNN.Momentum

var MICRO = false

type Loader interface {
	Load() (CnnDataset, error)
}

type CellCNN struct{ valid bool }

var CellCnnLoader Loader = CellCNN{false}

var CellCnnValidLoader Loader = CellCNN{true}

func (c CellCNN) Load() (CnnDataset, error) {
	if c.valid {
		return LoadCellCnnValidData(DATA_FOLDER, NSAMPLES, NCELLS, NFEATURES, cellCNN.TypeData), nil
	} else {
		return LoadCellCnnTrainData(DATA_FOLDER, NSAMPLES, NCELLS, NFEATURES), nil
	}
}

func GetLoader() (Loader, error) {
	return CellCnnLoader, nil
}

func GetValidLoader() (Loader, error) {
	return CellCnnLoader, nil
}
