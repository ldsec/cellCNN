package decentralized

import (
	"errors"
	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"
	"go.dedis.ch/onet/v3/network"
	"strconv"
)

func init() {
	network.RegisterMessages(
		new(Query),
		new(NewEncryptedIterationMessage),
		new(ChildUpdatedEncryptedLocalWeightsMessage),
	)
}

// TrainingProtocolName is the registered name for the regression_encrypted training protocol.
const TrainingProtocolName = "TrainingProtocol"

// Messages
//______________________________________________________________________________________________________________________

// NewEncryptedIterationMessage is the message sent by the root node to start a new global iteration
type NewEncryptedIterationMessage struct {
	IterationNumber int
	GlobalWeights   []byte
	CtSizes         []*[]int
}

// ChildUpdatedEncryptedLocalWeightsMessage is the message sent by the children to update the global weights
type ChildUpdatedEncryptedLocalWeightsMessage struct {
	ChildLocalWeights [][]byte
	Level             int64
	Pow2scale         int64
	Degree            int64
}

// Structs
//______________________________________________________________________________________________________________________
type newEncryptedIterationAnnouncementStruct struct {
	*onet.TreeNode
	NewEncryptedIterationMessage
}

// ChildUpdatedEncryptedLocalWeightsStruct is the message sent by the children to update the global weights
type ChildUpdatedEncryptedLocalWeightsStruct struct {
	*onet.TreeNode
	ChildUpdatedEncryptedLocalWeightsMessage
}

// TrainingProtocol keeps the state of the protocol
type TrainingProtocol struct {
	*onet.TreeNodeInstance

	// Root Channel
	WaitChannel chan struct{}

	AnnouncementChannel      chan newEncryptedIterationAnnouncementStruct
	ChildLocalWeightsChannel chan []ChildUpdatedEncryptedLocalWeightsStruct

	Debug bool
}

// NewTrainingProtocol initializes the protocol instance.
func NewTrainingProtocol(n *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
	pap := &TrainingProtocol{
		TreeNodeInstance:    n,
		WaitChannel:         make(chan struct{}, 1),
	}

	err := pap.RegisterChannel(&pap.AnnouncementChannel)
	if err != nil {
		return nil, errors.New("couldn't register announcement channel: " + err.Error())
	}

	err = pap.RegisterChannel(&pap.ChildLocalWeightsChannel)
	if err != nil {
		return nil, errors.New("couldn't register child-error channel: " + err.Error())
	}

	return pap, nil
}

// Start is called at the root to begin the execution of the protocol.
func (p *TrainingProtocol) Start() error {

}

// Dispatch is called at each node and handle incoming messages.
func (p *TrainingProtocol) Dispatch() error {
	defer p.Done()

	// Wait for the initialization of the weights (this is done in the Start())
	if p.IsRoot() {
		<-p.WaitChannel
	}

	if p.Debug {
		log.Lvl2("ProtoIter: "+strconv.Itoa(p.IterationNumber)+", "+p.ServerIdentity().String()+", NEW GLOBAL WEIGHTS:", p.GlobalWeights)
	}


}
