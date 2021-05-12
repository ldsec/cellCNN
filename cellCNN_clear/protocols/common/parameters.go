package common

// cellCNN parameters
const NCELLS = 200
const NFEATURES = 37
const NSAMPLES = 2000
const NSAMPLES_DIST = 400
const NCLASSES = 2
const NFILTERS = 7
const DATA_FOLDER = "../../data/cellCNN/original/"
const SPLIT_DATA_FOLDER = "../../data/cellCNN/split-original/"
const ApproxInterval = 3.
const testAllCell = 5652
const BATCH_SIZE = 40
const LEARN_RATE = 0.01
const MOMENTUM = 0.5

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
