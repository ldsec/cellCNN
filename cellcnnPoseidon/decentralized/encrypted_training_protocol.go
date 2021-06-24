package decentralized

import (
	"errors"
	"fmt"
	"runtime"
	"time"

	"github.com/ldsec/cellCNN/cellCNNClear/protocols/common"
	"github.com/ldsec/cellCNN/cellcnnPoseidon/centralized"
	"github.com/ldsec/cellCNN/cellcnnPoseidon/utils"
	"github.com/ldsec/lattigo/v2/ckks"
	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"
	"go.dedis.ch/onet/v3/network"
)

func init() {
	network.RegisterMessages(
		new(ChildUpdatedLocalGradientsMessage),
		new(NewEncryptedIterationMessage),
		new(SyncMessage),
	)
}

// NNEncryptedProtocolName is the registered name for the neural_network_encrypted training protocol.
const NNEncryptedProtocolName = "NNEncryptedProtocol"

// Messages
//______________________________________________________________________________________________________________________

// NewEncryptedIterationMessage is the message sent by the root node to start a new global iteration
type NewEncryptedIterationMessage struct {
	IterationNumber int
	GlobalWeights   [][]byte // first n-1 are conv filters, last one is dense weight
}

// ChildUpdatedLocalGradientsMessage contains the gradients to be aggregated
type ChildUpdatedLocalGradientsMessage struct {
	ChildUpdatedLocalGradients [][]byte // first n-1 are conv filters, last one is dense weight
	CurrentSize                int
	// SizeCiphers                []*[]int
}

// SyncMessage plain message struct used only for synchronization
type SyncMessage struct {
	Sync struct{}
}

// Structs
//______________________________________________________________________________________________________________________
type newEncryptedIterationStruct struct {
	*onet.TreeNode
	NewEncryptedIterationMessage
}

type childUpdatedLocalGradientsStruct struct {
	*onet.TreeNode
	ChildUpdatedLocalGradientsMessage
}

// SyncStruct onet wrapper of SyncMessage
type SyncStruct struct {
	*onet.TreeNode
	SyncMessage SyncMessage
}

// Protocol
//______________________________________________________________________________________________________________________

// SyncProtocol defines the messages exchanges to synchronise nodes (especially useful for experiments)
func SyncProtocol(n *onet.TreeNodeInstance, syncChannel chan SyncStruct) error {
	if n.IsRoot() {
		for i := 0; i < len(n.Roster().List)-1; i++ {
			<-syncChannel
		}
		n.Broadcast(&SyncMessage{Sync: struct{}{}})
	}
	if !n.IsRoot() {
		if !n.IsLeaf() {
			if err := n.SendToChildren(&SyncMessage{Sync: struct{}{}}); err != nil {
				return err
			}
		}
		if err := n.SendTo(n.Root(), &SyncMessage{Sync: struct{}{}}); err != nil {
			return err
		}
		<-syncChannel
	}
	return nil
}

// NNEncryptedProtocol keeps the state of the protocol
type NNEncryptedProtocol struct {
	*onet.TreeNodeInstance

	Version string

	// Protocol feedback channel
	FeedbackChannel chan [][]byte

	// Root Channel
	WaitChannel chan struct{}
	Sync        bool

	// Protocol communication channels
	NewEncryptedIterationChannel      chan newEncryptedIterationStruct
	ChildUpdatedLocalGradientsChannel chan []childUpdatedLocalGradientsStruct
	SyncChannel                       chan SyncStruct

	model *centralized.CellCNN

	MaxIterations   int
	IterationNumber int

	CryptoParams *utils.CryptoParams

	// Number of input, hidden and output nodes
	CellCNNSettings *utils.CellCnnSettings

	// currently not used, using randomly generated data from test
	TrainSet *common.CnnDataset

	LearningRate float64

	BatchSize int

	ApproximationFunction string
	ApproximationDegree   int
	Interval              []float64
	Coeffs                []float64

	evaluator ckks.Evaluator
	encoder   ckks.Encoder
}

// NewNNEncryptedProtocol initializes the protocol instance.
func NewNNEncryptedProtocol(n *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
	pap := &NNEncryptedProtocol{
		TreeNodeInstance: n,
		FeedbackChannel:  make(chan [][]byte),
		WaitChannel:      make(chan struct{}),
	}

	err := pap.RegisterChannel(&pap.NewEncryptedIterationChannel)
	if err != nil {
		return nil, errors.New("couldn't register <NewEncryptedIterationChannel> channel: " + err.Error())
	}

	err = pap.RegisterChannel(&pap.ChildUpdatedLocalGradientsChannel)
	if err != nil {
		return nil, errors.New("couldn't register <ChildUpdatedLocalGradientsChannel> channel: " + err.Error())
	}

	err = pap.RegisterChannel(&pap.SyncChannel)
	if err != nil {
		return nil, errors.New("couldn't register <SyncChannel> channel: " + err.Error())
	}

	return pap, nil
}

// Start is called at the root to begin the execution of the protocol.
func (p *NNEncryptedProtocol) Start() error {
	// p.Debug.ServerID = p.ServerIdentity().String()
	// log.Lvl2("[NN_START]", p.ServerIdentity(), " started a Neural Network Protocol")

	//CN_1 sends the initial weights to initiate the process
	weightsToSend := p.model.GetWeightsBinary()

	newEncryptedIterationMessage := NewEncryptedIterationMessage{p.IterationNumber, weightsToSend}

	// Unlock channel (root node can continue with Dispatch)
	if !p.Sync {
		p.WaitChannel <- struct{}{}
	}

	log.Lvl3("GlobalWeights to Send (bytes):", len(newEncryptedIterationMessage.GlobalWeights))
	if err := p.SendToChildren(&newEncryptedIterationMessage); err != nil {
		return err
	}

	fmt.Println("Start")

	return nil
}

// Dispatch is called at each node and handles incoming messages.
func (p *NNEncryptedProtocol) Dispatch() error {
	defer p.Done()

	fmt.Println("Dispatch")

	var err error

	// to syncronize
	if p.Sync {
		err := SyncProtocol(p.TreeNodeInstance, p.SyncChannel)
		if err != nil {
			return err
		}
		p.Sync = false
	} else {
		// Wait for the initialization of the weights (this is done in the Start())
		if p.IsRoot() {
			<-p.WaitChannel
		}
	}

	t1 := time.Now()

	newEncryptedIterationMessage := NewEncryptedIterationMessage{}
	for p.IterationNumber < p.MaxIterations { // protocol iterations
		// 1. Announcement phase
		finished := false

		// if it is not the root each node gets and forwards the general updated weights down
		if !p.IsRoot() {
			newEncryptedIterationMessage, err = p.newEncryptedIterationAnnouncementPhase()
			if err != nil {
				return err
			}

			// receive iteration_number and weights
			p.IterationNumber = newEncryptedIterationMessage.IterationNumber

			//need to check as new number is part of the message for non root nodes
			finished = p.IterationNumber >= p.MaxIterations

			p.model.LoadWeightsBinary(newEncryptedIterationMessage.GlobalWeights)

			if p.Index() == 1 {
				valuesTest := p.model.GetEncoder().Decode(ckks.NewDecryptor(p.CryptoParams.Params, p.CryptoParams.AggregateSk).DecryptNew(p.model.GetWeights()[0]), p.CryptoParams.Params.LogSlots())
				fmt.Printf("######## node index: %v, inital weights in round %v: %v\n########\n",
					p.Index(), p.IterationNumber,
					valuesTest[0:4],
				)
			}
		}

		fmt.Println("enter iteration: ", p.IterationNumber, p.IsRoot())

		if !finished {
			// 2. Aggregation of local gradients at level 2
			gradAgg, err := p.ascendingUpdateEncryptedGeneralModelPhase()
			fmt.Println("one client finish training one iteration")
			if err != nil {
				return err
			}

			if p.IsRoot() {
				// if p.Debug.Print {
				// 	log.Lvl2("ProtoIter: "+strconv.Itoa(p.IterationNumber)+", "+p.ServerIdentity().String()+", NEW WEIGHTS:", libspindle.DecryptMultipleFloat(p.CryptoParams, p.Weights[0][0], 0)[p.InitalGapSize+1:p.InitalGapSize+3])
				// }
				print(">>>>>> iteration: ", p.IterationNumber)
				p.IterationNumber = p.IterationNumber + 1

				if p.model.FisrtMomentum() {
					// if first momentum, btp scaled_g to level 9 and kept as vt
					gradAgg.DummyBootstrapping(p.encoder, p.CryptoParams.Params, p.CryptoParams.AggregateSk)
					p.model.UpdateMomentum(gradAgg)
				} else {
					// else, compute scaled_m at level 8 and get momentumed scaled at level 2
					gradAgg = p.model.ComputeScaledGradientWithMomentum(gradAgg, p.CellCNNSettings, p.CryptoParams.Params, p.model.GetEvaluator(), p.encoder, momentum)
					gradAgg.DummyBootstrapping(p.encoder, p.CryptoParams.Params, p.CryptoParams.AggregateSk)
					p.model.UpdateMomentum(gradAgg)
				}

				// bootstrap before global update
				p.UpdateRootWeights(gradAgg)

				// send updated weights down the tree
				weightsToSend := p.model.GetWeightsBinary()

				newIterationMessage := NewEncryptedIterationMessage{p.IterationNumber, weightsToSend}

				if err := p.SendToChildren(&newIterationMessage); err != nil {
					return err
				}
			}
		}
		runtime.GC()
	}

	t2 := time.Since(t1).Seconds()

	// 3. Results reporting
	if p.IsRoot() {
		fmt.Printf("+++++++ Time for %v protocol rounds: %v, avg: %v\n ++++++++", p.MaxIterations, t2, t2/float64(p.MaxIterations))
		p.FeedbackChannel <- p.model.GetWeightsBinary()
	}
	/*
		Some micro benchmarks
		rounds: 3
		batchsize: 5
		filters: 6

		gen rotation keys: ~25 seconds

		nodes	makers	cells	Time(each round)	communication(whole in bytes)
		3		24		50		62.490963			25166034
		5		24		50		199.61045313333332	50332068
		3		24 		100		81.7420632			25166034
		3 		48		50		116.1745403			25166034

	*/

	return nil
}

// Announce new_iteration down the tree
func (p *NNEncryptedProtocol) newEncryptedIterationAnnouncementPhase() (NewEncryptedIterationMessage, error) {
	// wait for the message from the root to start the protocol
	newEncryptedIterationMessage := <-p.NewEncryptedIterationChannel

	// if it is not leaf it propagates the message to the its children (in a tree-like way)
	if !p.IsLeaf() {
		log.Lvl3("GlobalWeights to Send (bytes):", len(newEncryptedIterationMessage.GlobalWeights))
		if err := p.SendToChildren(&newEncryptedIterationMessage.NewEncryptedIterationMessage); err != nil {
			return NewEncryptedIterationMessage{}, err
		}
	}
	return newEncryptedIterationMessage.NewEncryptedIterationMessage, nil
}

func (p *NNEncryptedProtocol) ascendingUpdateEncryptedGeneralModelPhase() (*centralized.Gradients, error) {
	// retrieve data points
	var gradients *centralized.Gradients
	var err error
	messageSize := 0

	if !p.IsRoot() {
		gradients, err = p.localIteration(p.model.GetEvaluator())
		if err != nil {
			return nil, err
		}
	}

	// gradients for nn
	var aggChild *centralized.Gradients
	if !p.IsLeaf() {
		// this reads all the gradients that were sent by all the children
		for i, v := range <-p.ChildUpdatedLocalGradientsChannel {
			if i == 0 {
				aggChild = new(centralized.Gradients)
				aggChild.NewGradient(v.ChildUpdatedLocalGradients)
			} else {
				aggChild.Aggregate(v.ChildUpdatedLocalGradients, p.model.GetEvaluator())
			}
			messageSize += v.CurrentSize
			messageSize += utils.SizeOf2DimSlice(v.ChildUpdatedLocalGradients)
		}

		if !p.IsRoot() {
			// add sum of children gradients with own local gradients
			aggChild.Aggregate(gradients, p.model.GetEvaluator())
		}

		if p.IsRoot() {
			fmt.Printf("++++++++++ Root Count the communication in bytes: %v +++++++++++++\n", messageSize)
		}

	} else {
		aggChild = gradients
		messageSize = 0
	}

	fmt.Printf("########\nIs Root: %v, index: %v\nChecking the values of the local agg gradients: %v\n########\n",
		p.IsRoot(), p.Index(),
		aggChild.GetPlaintext(0, []int{0, 1, 2}, p.CryptoParams.Params, p.model.GetEncoder(), ckks.NewDecryptor(p.CryptoParams.Params, p.CryptoParams.AggregateSk)),
	)

	// send the aggregated gradients up
	if !p.IsRoot() {
		data := aggChild.GetGradientBinary()
		log.Lvl3("Gradients to Send (bytes):", len(data)*len(data[0]))
		if err := p.SendToParent(&ChildUpdatedLocalGradientsMessage{ChildUpdatedLocalGradients: data, CurrentSize: messageSize}); err != nil {
			return nil, err
		}
	}
	return aggChild, nil
}

func (p *NNEncryptedProtocol) localIteration(eval ckks.Evaluator) (*centralized.Gradients, error) {
	// local node does not keep a momentum
	isMomentum := false

	// make a new batch
	X, _, y := utils.GetRandomBatch(p.TrainSet, p.BatchSize, p.CryptoParams.Params, p.encoder, p.CellCNNSettings)

	// batch forward and backward
	preds, _, _ := p.model.BatchProcessing(X, y, isMomentum)

	// get scaled gradients
	grad := p.model.GetGradients()

	valuesTest := p.model.GetEncoder().Decode(ckks.NewDecryptor(p.CryptoParams.Params, p.CryptoParams.AggregateSk).DecryptNew(preds[0]), p.CryptoParams.Params.LogSlots())
	fmt.Printf("######## node index: %v, preds: %v\n########\n", p.Index(), valuesTest[:10])

	return grad, nil
}

// UpdateRootWeights updates global model based on the aggregated gradients and bootstraps
func (p *NNEncryptedProtocol) UpdateRootWeights(gradientsAggr *centralized.Gradients) {
	//update at root
	p.model.UpdateWithGradients(gradientsAggr)
}
