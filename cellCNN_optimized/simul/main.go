package main

import (
	"fmt"
	"github.com/BurntSushi/toml"
	 "github.com/ldsec/cellCNN/cellCNN_optimized"
	"github.com/ldsec/cellCNN/cellCNN_optimized/decentralized"
	"math/rand"
	"time"

	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"

	"github.com/ldsec/lattigo/v2/ckks"
)

func init() {
	onet.SimulationRegister("CellCNN", NewCellCNNSimulation)
}

// CellCNNSimulation stores the state of a simulation
type CellCNNSimulation struct {
	onet.SimulationBFTree

	Debug bool

	TrainEncrypted bool
	Deterministic  bool

	Path            string
	PartyDataSize   int

	Epoch 		int
	Samples 	int
	Cells 		int
	Features 	int
	Filters 	int
	Classes 	int

	PathCryptoFiles string
	CkksParams   	*ckks.Parameters
	CryptoParams 	*cellCNN.CryptoParams
}

// NewCellCNNSimulation is the simulation instance constructor
func NewCellCNNSimulation(config string) (onet.Simulation, error) {
	sim := &CellCNNSimulation{}
	if err := toml.Unmarshal([]byte(config), sim); err != nil {
		return nil, err
	}

	if !sim.Deterministic{
		rand.Seed(time.Now().Unix())
	}
	aux := cellCNN.GenParams()
	sim.CkksParams = &aux

	return sim, nil
}

// Setup initializes the simulation
func (sim *CellCNNSimulation) Setup(dir string, hosts []string) (*onet.SimulationConfig, error) {
	sc := &onet.SimulationConfig{}
	sim.CreateRoster(sc, hosts, 2000)
	err := sim.CreateTree(sc)
	if err != nil {
		return nil, err
	}
	return sc, nil
}

// Node registers a KeySwitchSimulation for every node
func (sim *CellCNNSimulation) Node(config *onet.SimulationConfig) error {
	if _, err := config.Server.ProtocolRegister("NewCellCNNSimul",
		func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
			return NewCellCNNSimul(tni, sim)
		}); err != nil {
		return fmt.Errorf("error while registering <RegEncSimulation>:" + err.Error())
	}

	return sim.SimulationBFTree.Node(config)
}

// Run starts the simulation of the protocol and measures its runtime.
func (sim *CellCNNSimulation) Run(config *onet.SimulationConfig) error {
	log.SetDebugVisible(2)

	for round := 0; round < sim.Rounds; round++ {
		log.Lvl3("Starting round", round)

		rootInstance, err := config.Overlay.CreateProtocol("NewCellCNNSimul", config.Tree, onet.NilServiceID)
		if err != nil {
			return err
		}

		protocol := rootInstance.(*decentralized.TrainingProtocol)
		feedback := protocol.FeedbackChannel

		go protocol.Start()
		<-feedback
	}
	return nil
}

// NewCellCNNSimul is a simulation specific protocol instance constructor that injects test data.
func NewCellCNNSimul(tni *onet.TreeNodeInstance, sim *CellCNNSimulation) (onet.ProtocolInstance, error) {
	pi, err := decentralized.NewTrainingProtocol(tni)
	if err != nil {
		return nil, err
	}
	protocol := pi.(*decentralized.TrainingProtocol)

	vars := decentralized.InitCellCNNVars{
		Path:           sim.Path,
		PartyDataSize:  sim.PartyDataSize/tni.Tree().Size(),
		TrainEncrypted: sim.TrainEncrypted,
		Epochs:         sim.Epoch * cellCNN.Samples / cellCNN.BatchSize,
		Samples:        sim.Samples,
		Cells:          sim.Cells,
		Features:       sim.Features,
		Filters:        sim.Filters,
		Classes:        sim.Classes,
		Debug:          sim.Debug,
	}

	cryptoParamsList := cellCNN.ReadOrGenerateCryptoParams(1, sim.CkksParams, sim.PathCryptoFiles)
	protocol.InitVars(cryptoParamsList[0], sim.CkksParams, vars)

	// 1) Load Data
	protocol.XTrain, protocol.YTrain, _, _ = protocol.LoadData()


	return protocol, nil

}
