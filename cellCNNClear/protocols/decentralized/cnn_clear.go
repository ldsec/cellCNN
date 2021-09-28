package decentralized

import (
	"errors"
	"github.com/ldsec/cellCNN/cellCNNClear/layers"
	"github.com/ldsec/cellCNN/cellCNNClear/protocols/centralized"
	"github.com/ldsec/cellCNN/cellCNNClear/protocols/common"
	"github.com/ldsec/cellCNN/cellCNNClear/utils"
	libunlynx "github.com/ldsec/unlynx/lib"
	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"
	"go.dedis.ch/onet/v3/network"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"strconv"
)

// Debug is used to toggle on debug prints
type Debug struct {
	Print    bool
	ServerID string
}

const CnnClearProtocolName = "CNNClearProtocol"
const TIME = true
const display_freq = 10

func init() {
	network.RegisterMessage(ChildLocalUpdateMessage{})
	network.RegisterMessage(IterationMessage{})
	if _, err := onet.GlobalProtocolRegister(CnnClearProtocolName, NewCnnClearProtocol); err != nil {
		log.Fatal("Failed to register the <BatchGradProtocol>")
	}
}

// Messages
//______________________________________________________________________________________________________________________

type IterationMessage struct {
	IterationNumber int
	Weights         common.WeightsVector
}

// ChildUpdatedDataMessage contains one node's weights update.
type ChildLocalUpdateMessage struct {
	ChildLocalDeltas common.WeightsVector
}

// Structs
//______________________________________________________________________________________________________________________
type IterationAnnouncement struct {
	*onet.TreeNode
	IterationMessage
}

type ChildLocalUpdate struct {
	*onet.TreeNode
	ChildLocalUpdateMessage
}

// Protocol
//______________________________________________________________________________________________________________________

// CnnClearProtocol runs an unencrypted decentralized Convolutional Neural Network protocol.
type CnnClearProtocol struct {
	*onet.TreeNodeInstance

	// Protocol feedback channel
	FeedbackChannel chan common.WeightsVector

	// Root Channel
	WaitChannel chan int

	// Protocol communication channels
	AnnouncementChannel     chan IterationAnnouncement
	ChildLocalDeltasChannel chan []ChildLocalUpdate

	// weights
	Weights common.WeightsVector

	MaxIterations   int
	IterationNumber int

	Debug Debug

	X []*mat.Dense
	Y []float64

	// layers
	Conv  layers.Conv1D
	Pool  layers.Pool
	Dense layers.Dense_n

	LearningRate float64
	momentum     float64
	NFilters     int
	FirstMoment  bool

	//deltas for batch averaging
	EOut, EHidden []common.WeightsVector
	// Last change in weights for momentum
	ChangeMoment common.WeightsVector

	BatchSize            int
	LocalIterationNumber int
}

// NewCnnClearProtocol initializes the protocol instance.
func NewCnnClearProtocol(n *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
	pap := &CnnClearProtocol{
		TreeNodeInstance: n,
		FeedbackChannel:  make(chan common.WeightsVector),
		WaitChannel:      make(chan int),
		IterationNumber:  0,
	}

	err := pap.RegisterChannel(&pap.AnnouncementChannel)
	if err != nil {
		return nil, errors.New("couldn't register announcement channel: " + err.Error())
	}

	err = pap.RegisterChannel(&pap.ChildLocalDeltasChannel)
	if err != nil {
		return nil, errors.New("couldn't register child-error channel: " + err.Error())
	}

	return pap, nil
}

// Start is called at the root to begin the execution of the protocol.
func (p *CnnClearProtocol) Start() error {
	p.Debug.ServerID = p.ServerIdentity().String()
	log.Lvl2("[BGP_START]", p.ServerIdentity(), " started a Batch Gradient Protocol")

	iterationZero := libunlynx.StartTimer(p.Name() + "_IterationZero")

	//CN_1 sends the initial weights to initiate the process
	newIterationMessage := IterationMessage{p.IterationNumber, p.Weights}

	libunlynx.EndTimer(iterationZero)

	// Unlock channel (root node can continue with Dispatch)
	p.WaitChannel <- 1
	if err := p.SendToChildren(&newIterationMessage); err != nil {
		log.Fatal("Error sending <ForwardPassMessage>: ", p.ServerIdentity().String(), " ", err)
	}
	return nil
}

// Dispatch is called at each node and handle incoming messages.
func (p *CnnClearProtocol) Dispatch() error {
	defer p.Done()

	// Wait for the initialization of the weights (this is done in the Start())
	if p.IsRoot() {
		<-p.WaitChannel
	}

	newIterationMessage := IterationMessage{}

	for p.IterationNumber < p.MaxIterations { // protocol iterations
		// 1. Forward Pass announcement phase
		finished := false
		if !p.IsRoot() {
			// if it is not the root each node gets and forwards the general updated weights down
			newIterationMessage = p.newIterationAnnouncementPhase()
			p.IterationNumber = newIterationMessage.IterationNumber

			// need to check as new number is part of the message for non root nodes
			finished = p.IterationNumber >= p.MaxIterations

			// receive weights
			p.Weights = newIterationMessage.Weights

			// precomputation for non CN_1 nodes
			if p.IterationNumber == 0 {
				p.iterationZero()
			}
		}

		if !finished {
			// get results of local iterations
			totalChange := p.ascendingUpdateGeneralModelPhase()
			if p.IsRoot() {

				updateWeights := libunlynx.StartTimer("UpdateWeights")

				if p.IterationNumber%10 == 0 {
					log.Lvl2("ProtoIter: " + strconv.Itoa(p.IterationNumber) + ", " + p.ServerIdentity().String())
				}
				p.IterationNumber = p.IterationNumber + 1

				//update weights
				for i := range p.Weights {

					p.Weights[i].Sub(p.Weights[i], totalChange[i])

					if p.momentum > 0 {
						if p.FirstMoment == false {
							p.ChangeMoment[i] = totalChange[i]
						} else {
							p.ChangeMoment[i].Scale(p.momentum, p.ChangeMoment[i])
							p.ChangeMoment[i].Add(p.ChangeMoment[i], totalChange[i])
						}
						p.Weights[i].Sub(p.Weights[i], p.ChangeMoment[i])
					} else {
						p.Weights[i].Sub(p.Weights[i], totalChange[i])
					}
				}
				p.FirstMoment = true

				// send updated weights
				newIterationMessage := IterationMessage{p.IterationNumber, p.Weights}

				libunlynx.EndTimer(updateWeights)

				if err := p.SendToChildren(&newIterationMessage); err != nil {
					log.Fatal("Error sending <ForwardPassMessage>: ", p.ServerIdentity().String(), " ", err)
				}
			}
		}
	}

	// 3. Response reporting
	if p.IsRoot() {
		p.FeedbackChannel <- p.Weights
	}

	return nil
}

// Announce forwarding down the tree.
func (p *CnnClearProtocol) newIterationAnnouncementPhase() IterationMessage {
	// wait for the message from the root to start the protocol
	newIterationMessage := <-p.AnnouncementChannel
	// if it is not leaf it propagates the message to the its children (in a tree-like way)

	if !p.IsLeaf() {
		if err := p.SendToChildren(&newIterationMessage.IterationMessage); err != nil {
			log.Fatal("Error sending <ForwardPassMessage>: ", p.ServerIdentity().String())
		}
	}
	return newIterationMessage.IterationMessage
}

// first iteration
func (p *CnnClearProtocol) iterationZero() {
	//
}

// Local updates, Results pushing up the tree containing aggregation results.
func (p *CnnClearProtocol) ascendingUpdateGeneralModelPhase() common.WeightsVector {
	//Take a random batch:
	newBatch := make([]*mat.Dense, p.BatchSize)
	newBatchLabels := make([]float64, p.BatchSize)
	for j := 0; j < len(newBatch); j++ {
		randi := rand.Intn(len(p.X))
		newBatch[j] = p.X[randi]
		newBatchLabels[j] = p.Y[randi]
	}

	totalChange := p.localIteration(newBatch, newBatchLabels)

	combineTimer := libunlynx.StartTimer(p.Name() + "_Combine")

	// deltas for nn (for each layer)
	localUpdatedDeltas := make(common.WeightsVector, 0)
	if !p.IsLeaf() {
		// this reads all the weights that were sent by all the children
		for i, v := range <-p.ChildLocalDeltasChannel {
			if i == 0 {
				localUpdatedDeltas = v.ChildLocalDeltas
			} else {
				tmp := v.ChildLocalDeltas
				// add the weights for each layer
				for lw := range localUpdatedDeltas {
					localUpdatedDeltas[lw].Add(tmp[lw], localUpdatedDeltas[lw])
				}
			}
		}

		// add sum of children deltas with own local deltas
		for lw := range localUpdatedDeltas {
			localUpdatedDeltas[lw].Add(totalChange[lw], localUpdatedDeltas[lw])
		}
	} else {
		localUpdatedDeltas = totalChange
	}

	// send the updated weights (containing children) up
	if !p.IsRoot() {
		if err := p.SendToParent(&ChildLocalUpdateMessage{localUpdatedDeltas}); err != nil {
			log.Fatal("Error sending <ChildErrorBytesMessage>: ", p.ServerIdentity().String())
		}
	}

	libunlynx.EndTimer(combineTimer)

	return localUpdatedDeltas
}

func (p *CnnClearProtocol) localIteration(batch []*mat.Dense, labels []float64) common.WeightsVector {

	localIterationTimer := libunlynx.StartTimer(p.Name() + "_LocalIteration")

	var totalChange common.WeightsVector
	for i := 0; i < p.LocalIterationNumber; i++ {
		// Feed Forward
		output := p.feedForward(batch, i == 0)
		// Back Propagation
		totalChange = p.backPropagation(labels, output)
	}
	p.Weights = []*mat.Dense{p.Conv.GetWeights(), p.Dense.GetWeights()}

	libunlynx.EndTimer(localIterationTimer)

	return totalChange
}

func (p *CnnClearProtocol) feedForward(batch []*mat.Dense, firstLocalIteration bool) *mat.Dense {
	var output *mat.Dense
	var out1 []*mat.Dense
	var out2 *mat.Dense
	if firstLocalIteration {
		out1 = p.Conv.Forward(batch, p.Weights[0])
		out2 = p.Pool.Forward(out1)
		output = p.Dense.Forward(out2, p.Weights[1])
	} else {
		out1 = p.Conv.Forward(batch, nil)
		out2 = p.Pool.Forward(out1)
		output = p.Dense.Forward(out2, nil)
	}
	return output
}

func (p *CnnClearProtocol) backPropagation(labels []float64, output *mat.Dense) common.WeightsVector {
	deltas := make(common.WeightsVector, len(p.Weights))

	var gradient mat.Dense

	computeGrad := func(i int, j int, v float64) float64 {
		return centralized.ComputeGradient(i, j, v, labels)
	}

	gradient.Apply(computeGrad, output)

	var delta2 *mat.Dense
	delta2, deltas[1] = p.Dense.Backward(&gradient, p.LearningRate, p.momentum)
	delta1 := p.Pool.Backward(delta2)
	deltas[0] = p.Conv.Backward(delta1, p.LearningRate, p.momentum)

	return deltas
}

func (p *CnnClearProtocol) InitWeights() {
	_, nmarkers := p.X[0].Dims()
	p.ChangeMoment = []*mat.Dense{mat.NewDense(nmarkers, p.Conv.Nfilters, utils.WeightsInit(nmarkers*p.Conv.Nfilters, float64(nmarkers))),
		mat.NewDense(p.Conv.Nfilters, p.Dense.Nclasses, utils.WeightsInit(p.Conv.Nfilters*p.Dense.Nclasses, float64(p.Conv.Nfilters)))}
	p.Weights = []*mat.Dense{mat.NewDense(nmarkers, p.Conv.Nfilters, utils.WeightsInit(nmarkers*p.Conv.Nfilters, float64(nmarkers))),
		mat.NewDense(p.Conv.Nfilters, p.Dense.Nclasses, utils.WeightsInit(p.Conv.Nfilters*p.Dense.Nclasses, float64(p.Conv.Nfilters)))}
}
