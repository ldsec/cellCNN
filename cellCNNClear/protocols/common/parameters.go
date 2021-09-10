package common

// cellCNN parameters
const NCELLS = 200
const NFEATURES = 37
const NSAMPLES = 1000
const NSAMPLES_DIST = 400

var NCLASSES = 2

const NFILTERS = 8
const DATA_FOLDER = "../../data/cellCNN/originalNK/"
const SPLIT_DATA_FOLDER = "../../data/cellCNN/splitNK/"
const ApproxInterval = 3.
const TESTALLCELL = 5652
const BATCH_SIZE = 20
const LEARN_RATE = 0.01
const MOMENTUM = 0.9

var MICRO = false

type Loader interface {
	Load() (CnnDataset, error)
}

type CellCNN struct{ valid bool }

var CellCnnLoader Loader = CellCNN{false}

var CellCnnValidLoader Loader = CellCNN{true}

func (c CellCNN) Load() (CnnDataset, error) {
	if c.valid {
		return LoadCellCnnValidData(DATA_FOLDER, NSAMPLES, NCELLS, NFEATURES), nil
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
