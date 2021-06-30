package main

import (
	"errors"
	"github.com/BurntSushi/toml"
	"github.com/ldsec/cellCNN/cellCNNClear/protocols/common"
	"github.com/ldsec/cellCNN/cellCNNClear/protocols/decentralized"
	"github.com/ldsec/unlynx/lib"
	"github.com/pelletier/go-toml"
	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"
)

func init() {
	onet.SimulationRegister("DecentralizedCellCNN", NewCnnSimulation)
}

// CnnSimulation stores the state of the simulation
type CnnSimulation struct {
	onet.SimulationBFTree
	Time    bool
	Mininet bool

	Seed  int64
	KFold uint

	datasetLoader common.Loader
	LearnRate     float64
	Momentum      float64
	NbrEpochs     int
	NbrLocalIter  int
	NodeBatchSize int
}

// Debug is used to toggle on debug prints
type Debug struct {
	Print    bool
	ServerID string
}

// NewCnnSimulation is the simulation instance constructor
func NewCnnSimulation(config string) (onet.Simulation, error) {
	var err error

	sim := &CnnSimulation{}
	if err = toml.Unmarshal([]byte(config), sim); err != nil {
		return nil, err
	}

	sim.Seed = 0

	if sim.datasetLoader, err = common.GetLoader(); err != nil {
		return nil, err
	}

	return sim, nil
}

// Setup initializes the simulation.
func (sim *CnnSimulation) Setup(dir string, hosts []string) (*onet.SimulationConfig, error) {

	sc := &onet.SimulationConfig{}
	sim.CreateRoster(sc, hosts, 2000)
	err := sim.CreateTree(sc)
	if err != nil {
		return nil, err
	}
	return sc, nil
}

// Node registers a RegSimul protocol
func (sim *CnnSimulation) Node(config *onet.SimulationConfig) error {
	if _, err := config.Server.ProtocolRegister("NewCnnProtocolSimul",
		func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
			return NewCnnProtocolSimul(tni, sim)
		}); err != nil {
		return errors.New("Error while registering <NewCnnSimul>:" + err.Error())
	}

	return sim.SimulationBFTree.Node(config)
}

// Run starts the simulation of the protocol and measures its runtime.
func (sim *CnnSimulation) Run(config *onet.SimulationConfig) error {
	log.SetDebugVisible(2)
	if sim.Time {
		libunlynx.TIME = true
	}

	for round := 0; round < sim.Rounds; round++ {
		log.Lvl2("Starting round", round)
		//time measurement
		roundTime := libunlynx.StartTimer("DecentralizedCNN(SIMULATION)")

		err, s := decentralized.RunCnnClearTest(nil, config.Overlay, config.Tree, !sim.Mininet, sim.Time,
			"NewCnnProtocolSimul", sim.KFold, sim.datasetLoader)
		log.LLvl1(s)
		if err != nil {
			return err
		}
		libunlynx.EndTimer(roundTime)
	}
	return nil
}

// NewCnnProtocolSimul is a simulation specific protocol instance constructor that injects test data.
func NewCnnProtocolSimul(tni *onet.TreeNodeInstance, sim *CnnSimulation) (onet.ProtocolInstance, error) {
	cnnProtocolSimul := libunlynx.StartTimer(tni.Name() + "_CnnProtocolSimul(PRE-COMPUTATION)")

	pi, err := decentralized.NewCnnClearProtocol(tni)
	if err != nil {
		return nil, err
	}
	protocol := pi.(*decentralized.CnnClearProtocol)

	// ##STEP 1: Load data
	trainData := common.LoadCellCnnTrainData()

	// ##STEP 2: Split data
	protocol.X, protocol.Y, protocol.MaxIterations = common.SplitData(trainData.X, trainData.Y, len(tni.Roster().List),
		tni.Index(), sim.NbrEpochs, sim.NbrLocalIter, sim.NodeBatchSize, tni.IsRoot())

	// ##STEP 3: InitRoot protocol training variables
	decentralized.InitCnnClearProtocolVars(protocol, sim.LearnRate, sim.Momentum, sim.NbrLocalIter, sim.NodeBatchSize)

	libunlynx.EndTimer(cnnProtocolSimul)

	protocol.Debug.Print = false
	return protocol, nil
}
