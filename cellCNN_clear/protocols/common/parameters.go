package common

// cellCNN parameters
const NCELLS = 200
const NFEATURES = 37
const NSAMPLES = 2000
const NCLASSES = 2
const NFILTERS = 6
const DATA_FOLDER = "../../data/cellCNN/normalized/"
const SPLIT_DATA_FOLDER = "../../data/cellCNN/split-normalized/"
const ApproxInterval = 3.

const BATCH_SIZE = 200
const LEARN_RATE = 0.1
const MOMENTUM = 0.9

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
