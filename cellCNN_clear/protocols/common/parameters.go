package common

// cellCNN parameters
const NCELLS = 200
const NFEATURES = 16
const NSAMPLES = 2250
const NSAMPLES_DIST = 1000
const NCLASSES = 3
const NFILTERS = 7
const DATA_FOLDER = "../../data/cellCNN/originalAML/"
const SPLIT_DATA_FOLDER = "../../data/cellCNN/split/"
const ApproxInterval = 3.
const testAllCell = 12440
const BATCH_SIZE = 100
const LEARN_RATE = 0.002
const MOMENTUM = 0.9
const MICRO = false

type Loader interface {
	Load() (CnnDataset, error)
}

type CellCNN struct{ valid bool }

var CellCnnLoader Loader = CellCNN{false}

var CellCnnValidLoader Loader = CellCNN{true}

func (c CellCNN) Load() (CnnDataset, error) {
	if c.valid {
		return LoadCellCnnValidData(), nil
	} else {
		return LoadCellCnnTrainData(), nil
	}
}

func GetLoader() (Loader, error) {
	return CellCnnLoader, nil
}

func GetValidLoader() (Loader, error) {
	return CellCnnLoader, nil
}
