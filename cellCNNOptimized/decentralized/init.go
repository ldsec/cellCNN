package decentralized

import (
	"github.com/ldsec/cellCNN/cellCNNOptimized"
	"github.com/ldsec/lattigo/v2/ckks"
)

type InitCellCNNVars struct {
	Path string

	TrainPlain     bool
	TrainEncrypted bool
	Deterministic  bool

	Epochs       int
	LocalSamples int

	Debug bool
}

func (p *TrainingProtocol) InitVars(cryptoParams *cellCNN.CryptoParams, params *ckks.Parameters, vars InitCellCNNVars) {

	// party creation
	p.CNNProtocol = cellCNN.NewCellCNNProtocol(*params)

	if vars.TrainEncrypted {
		// key generation
		p.CNNProtocol.SetSecretKey(cryptoParams.Sk)
		p.CNNProtocol.SetPublicKey(cryptoParams.Pk)
		cryptoParams.SetRotKeys(p.CNNProtocol.RotKeyIndex())
		p.CryptoParams = cryptoParams
		p.CNNProtocol.EvaluatorInit(cryptoParams.Rlk, cryptoParams.RotKs)
	}

	// init vars
	p.Path = vars.Path

	p.TrainPlain = vars.TrainPlain
	p.TrainEncrypted = vars.TrainEncrypted
	p.Deterministic = vars.Deterministic

	p.PrngInt = cellCNN.NewPRNTInt(vars.LocalSamples, vars.Deterministic)

	p.Epochs = vars.Epochs

	p.Debug = vars.Debug
}
