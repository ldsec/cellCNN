package main

import (
	"fmt"
	"github.com/BurntSushi/toml"
	"github.com/ldsec/cellCNN/cellCNN_optimized"
	"github.com/ldsec/cellCNN/cellCNN_optimized/decentralized"
	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"

	"github.com/ldsec/lattigo/v2/ckks"
)

func init() {
	onet.SimulationRegister("CellCNNBootstrap", NewCellCNNSimulation)
}

// CellCNNSimulation stores the state of a simulation
type CellCNNSimulation struct {
	onet.SimulationBFTree

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

	cellCNN.LogN = 16
	cellCNN.LogSlots = cellCNN.LogN-1
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
	if _, err := config.Server.ProtocolRegister("NewCellCNNBootSimul",
		func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
			return NewCellCNNBootSimul(tni, sim)
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

		rootInstance, err := config.Overlay.CreateProtocol("NewCellCNNBootSimul", config.Tree, onet.NilServiceID)
		if err != nil {
			return err
		}

		protocol := rootInstance.(*decentralized.BootstrapProtocol)
		feedback := protocol.FeedbackChannel

		go protocol.Start()
		<-feedback
	}
	return nil
}

// NewCellCNNBootSimul is a simulation specific protocol instance constructor that injects test data.
func NewCellCNNBootSimul(tni *onet.TreeNodeInstance, sim *CellCNNSimulation) (onet.ProtocolInstance, error) {
	cryptoParamsList := cellCNN.ReadOrGenerateCryptoParams(1, sim.CkksParams, sim.PathCryptoFiles + tni.ServerIdentity().Address.String() + "/paramsCKKS")
	pi, err := decentralized.NewBootstrapProtocol(tni, cryptoParamsList[0], nil)
	if err != nil {
		return nil, err
	}
	return pi, nil
}
