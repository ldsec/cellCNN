package decentralized

import (
	cellCNN "github.com/ldsec/cellCNN/cellCNN_optimized"
)

type InitCellCNNVars struct {
	Path			string
	PartyDataSize   int

	TrainEncrypted  bool
	Deterministic   bool
	
	Epochs          int

	Samples 		int
	Cells			int
	Features		int
	Filters			int
	Classes			int

	Debug 			bool
}

func (p *TrainingProtocol) InitVars(cryptoParams *cellCNN.CryptoParams, vars InitCellCNNVars) {

	// party creation
	p.CNNProtocol = cellCNN.NewCellCNNProtocol(*cryptoParams.Params)

	// key generation
	p.CNNProtocol.SetSecretKey(cryptoParams.Sk)
	p.CNNProtocol.SetPublicKey(cryptoParams.Pk)
	cryptoParams.SetRotKeys(p.CNNProtocol.RotKeyIndex())
	p.CryptoParams = cryptoParams
	p.CNNProtocol.EvaluatorInit(cryptoParams.Rlk, cryptoParams.RotKs)

	// init vars
	p.Path = vars.Path
	p.PartyDataSize = vars.PartyDataSize

	p.TrainEncrypted = vars.TrainEncrypted
	p.Deterministic = vars.Deterministic

	p.PrngInt = cellCNN.NewPRNTInt(vars.Samples, vars.Deterministic)

	p.Epochs = vars.Epochs

	p.Samples = vars.Samples
	p.Cells = vars.Cells
	p.Features = vars.Features
	p.Filters = vars.Filters
	p.Classes = vars.Classes

	p.Debug = vars.Debug
}

