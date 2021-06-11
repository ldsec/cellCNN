package decentralized

import (
	"errors"
	"fmt"
	"runtime"

	"github.com/ldsec/cellCNN/cellCNN_clear/protocols/common"
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
	GlobalSk        []byte
	GlobalPk        []byte
}

// ChildUpdatedLocalGradientsMessage contains the gradients to be aggregated
type ChildUpdatedLocalGradientsMessage struct {
	ChildUpdatedLocalGradients [][]byte // first n-1 are conv filters, last one is dense weight
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

	// Weights *ckks.Ciphertext // global initial weights

	model *centralized.CellCNN

	MaxIterations   int
	IterationNumber int

	CryptoParams *utils.CryptoParams

	// Debug libspindle.Debug

	// Number of input, hidden and output nodes
	CellCNNSettings *utils.CellCnnSettings

	// Activations for nodes
	// InputActivations                           []*ckks.Plaintext
	// HiddenActivations, HiddenBeforeActivations []libspindle.CipherMatrix
	// OutputActivations, OutBeforeActivations    libspindle.CipherVector

	// ElmanRNN contexts
	// Contexts [][]float64

	// X, Y [][]float64
	TrainSet *common.CnnDataset

	//deltas for batch averaging
	// EOut, EHidden [][]*ckks.Ciphertext
	// Last change in weights for momentum
	// DeltaOut, DeltaHidden [][]*ckks.Ciphertext

	// Mask, Mask2, Mask3, MaskLast, LearningRatePacked *ckks.Plaintext
	LearningRate float64

	BatchSize int

	ApproximationFunction string
	ApproximationDegree   int
	Interval              []float64
	Coeffs                []float64

	evaluator ckks.Evaluator
	encoder   ckks.Encoder

	// InitalGap       []float64 //put this gap for ANY encoded/encrypted vector to get rid of accumulated error
	// InitalGapSize   int
	// TotalGapBetween []int

	// NumColsPerCipher     int
	// NumTotalCiphers      int
	// NumColsLastCipher    int
	// IsLastDifferent      bool
	// InputActivationsLast []*ckks.Plaintext

	// BlockSize int
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
	weightsToSend := p.model.Marshall()

	// use common sk, pk across all nodes for test
	dsk, spk := p.CryptoParams.MarshalBinary()
	newEncryptedIterationMessage := NewEncryptedIterationMessage{p.IterationNumber, weightsToSend, dsk, spk}

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
			// p.model.Unmarshall(newEncryptedIterationMessage.GlobalWeights)
			//need to check as new number is part of the message for non root nodes
			finished = p.IterationNumber >= p.MaxIterations

			// test: if iter = 0, use common pk, sk
			// if p.IterationNumber == 0 {
			// 	p.CryptoParams.RetrieveCommonParams(newEncryptedIterationMessage.GlobalSk, newEncryptedIterationMessage.GlobalPk)
			// 	//only in root, and root will pass weights down the tree
			// 	p.model = centralized.NewCellCNN(p.CellCNNSettings, p.CryptoParams)
			// 	p.model.InitWeights(nil, nil, nil, nil)
			// 	p.model.Unmarshall(newEncryptedIterationMessage.GlobalWeights)
			// 	eval := p.model.InitEvaluator(p.CryptoParams, maxM1N2Ratio)
			// 	p.evaluator = eval
			// 	// use sk for bootstrapping
			// 	p.model.WithSk(p.CryptoParams.Sk)
			// } else {
			// 	p.model.Unmarshall(newEncryptedIterationMessage.GlobalWeights)
			// }

			p.model.Unmarshall(newEncryptedIterationMessage.GlobalWeights)

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
				p.IterationNumber = p.IterationNumber + 1

				if p.model.FisrtMomentum() {
					// if first momentum, btp scaled_g to level 9 and kept as vt
					gradAgg.Bootstrapping(p.encoder, p.CryptoParams.Params, p.CryptoParams.AggregateSk)
					p.model.UpdateMomentum(gradAgg)
				} else {
					// else, compute scaled_m at level 8 and get momentumed scaled at level 2
					gradAgg := p.model.ComputeScaledGradientWithMomentum(gradAgg, p.CellCNNSettings, p.CryptoParams.Params, p.model.GetEvaluator(), p.encoder, momentum)
					gradAgg.Bootstrapping(p.encoder, p.CryptoParams.Params, p.CryptoParams.AggregateSk)
					p.model.UpdateMomentum(gradAgg)
				}

				// bootstrap before global update
				p.UpdateRootWeights(gradAgg)

				// send updated weights down the tree
				weightsToSend := p.model.Marshall()

				newIterationMessage := NewEncryptedIterationMessage{p.IterationNumber, weightsToSend, nil, nil}

				if err := p.SendToChildren(&newIterationMessage); err != nil {
					return err
				}
			}
		}
		runtime.GC()
	}

	// 3. Results reporting
	if p.IsRoot() {
		p.FeedbackChannel <- p.model.Marshall()
	}

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
		}

		if !p.IsRoot() {
			// add sum of children gradients with own local gradients
			aggChild.Aggregate(gradients, p.model.GetEvaluator())
		}

	} else {
		aggChild = gradients
	}

	fmt.Printf("########\nIs Root: %v, index: %v\nChecking the values of the local agg gradients: %v\n########\n",
		p.IsRoot(), p.Index(),
		aggChild.GetPlaintext(0, []int{0, 1, 2}, p.CryptoParams.Params, p.model.GetEncoder(), ckks.NewDecryptor(p.CryptoParams.Params, p.CryptoParams.AggregateSk)),
	)

	// send the aggregated gradients up
	if !p.IsRoot() {
		data := aggChild.Marshall()
		log.Lvl3("Gradients to Send (bytes):", len(data)*len(data[0]))
		if err := p.SendToParent(&ChildUpdatedLocalGradientsMessage{ChildUpdatedLocalGradients: data}); err != nil {
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
	preds := p.model.BatchProcessing(X, y, isMomentum)

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

// // DecryptNNFinalWeights decrypt and extracts the cleartext weights for a 1-layer neural network model
// func (p *NNEncryptedProtocol) DecryptNNFinalWeights(weights libspindle.CipherMatrix) [][][]float64 {
// 	//write the decrypted weights into matrix (don't forget to skip the first initialGap slots)
// 	inputWeights := libspindle.Matrix(p.NInputs, p.NHiddens[0])
// 	outputWeights := libspindle.Matrix(p.NHiddens[0], p.NOutputs)
// 	inputWeightsTemp := make([]float64, 0)
// 	outputWeightsTemp := make([]float64, 0)

// 	//decrypt weights:
// 	for i := 0; i < p.NumTotalCiphers; i++ {
// 		inputWeightsTemp = append(inputWeightsTemp, libspindle.DecryptMultipleFloat(p.CryptoParams, weights[0][i], -1)...)
// 		outputWeightsTemp = append(outputWeightsTemp, libspindle.DecryptMultipleFloat(p.CryptoParams, weights[1][i], -1)...)
// 	}
// 	//calculate total gap at the end of each cipher appended for input weights:
// 	totalGapEnd := p.CryptoParams.GetSlots() - (p.InitalGapSize + (p.NumColsPerCipher * p.NInputs))
// 	//merge with next ciphers initial gap:
// 	totalGapEnd = totalGapEnd + p.InitalGapSize
// 	//input weights were column packed..
// 	indE := p.InitalGapSize
// 	flag := 0
// 	for i := 0; i < p.NHiddens[0]; i++ {
// 		for j := 0; j < p.NInputs; j++ {
// 			inputWeights[j][i] = inputWeightsTemp[indE]
// 			indE++
// 		}
// 		//jump flag
// 		flag++
// 		if flag == p.NumColsPerCipher {
// 			indE = indE + totalGapEnd
// 			flag = 0
// 		}
// 	}
// 	//output was row packed with a gap in between
// 	indE = p.InitalGapSize
// 	flag = 0
// 	for i := 0; i < p.NHiddens[0]; i++ {
// 		for j := 0; j < p.NOutputs; j++ {
// 			outputWeights[i][j] = outputWeightsTemp[indE]
// 			indE++
// 		}
// 		indE += p.NInputs - p.NOutputs
// 		flag++
// 		if flag == p.NumColsPerCipher {
// 			indE = indE + totalGapEnd
// 			flag = 0
// 		}
// 	}
// 	return [][][]float64{inputWeights, outputWeights}
// }

// // DecryptModel2Layer decrypt and extracts the cleartext weights for a 2-layer neural network model
// func (p *NNEncryptedProtocol) DecryptModel2Layer(weights libspindle.CipherMatrix) [][][]float64 {
// 	//decrypt weights:
// 	inputWeightsTemp := make([]float64, 0)
// 	outputWeightsTemp := make([]float64, 0)
// 	hiddenWeightsTemp := make([]float64, 0)

// 	//decrypt weights:
// 	for i := 0; i < p.NumTotalCiphers; i++ {
// 		inputWeightsTemp = append(inputWeightsTemp, libspindle.DecryptMultipleFloat(p.CryptoParams, p.Weights[0][i], -1)...)
// 		hiddenWeightsTemp = append(hiddenWeightsTemp, libspindle.DecryptMultipleFloat(p.CryptoParams, p.Weights[1][i], -1)...)
// 	}
// 	outputWeightsTemp = append(outputWeightsTemp, libspindle.DecryptMultipleFloat(p.CryptoParams, p.Weights[2][0], -1)...)

// 	//write the decrypted weights into matrix (don't forget to skip the first initialGap slots)
// 	inputWeights := libspindle.Matrix(p.NInputs, p.NHiddens[0])
// 	hiddenWeights := libspindle.Matrix(p.NHiddens[0], p.NHiddens[1])
// 	outputWeights := libspindle.Matrix(p.NHiddens[1], p.NOutputs)

// 	totalGapBetweenInput := 0
// 	totalGapBetweenHidden := 0
// 	totalGapEnd := 0
// 	if p.NHiddens[0] > p.NInputs {
// 		totalGapBetweenInput = p.NHiddens[0] - p.NInputs //put this gap between rows
// 		totalGapEnd = p.CryptoParams.GetSlots() - (p.InitalGapSize + (p.NumColsPerCipher * p.NHiddens[0]))

// 	}
// 	if p.NHiddens[0] < p.NInputs {
// 		totalGapBetweenHidden = p.NInputs - p.NHiddens[0] //put this gap between rows
// 		totalGapEnd = p.CryptoParams.GetSlots() - (p.InitalGapSize + (p.NumColsPerCipher * p.NInputs))
// 	}

// 	//calculate total gap at the end of each cipher appended for input weights:
// 	//merge with next ciphers initial gap:
// 	totalGapEnd = totalGapEnd + p.InitalGapSize
// 	//input weights were column packed with a gap in btw..
// 	indE := p.InitalGapSize
// 	flag := 0
// 	for i := 0; i < p.NHiddens[0]; i++ {
// 		for j := 0; j < p.NInputs; j++ {
// 			inputWeights[j][i] = inputWeightsTemp[indE]
// 			indE++
// 		}
// 		indE += totalGapBetweenInput
// 		//jump flag
// 		flag++
// 		if flag == (p.NumColsPerCipher) {
// 			indE = indE + totalGapEnd
// 			flag = 0
// 		}
// 	}
// 	//output was col packed
// 	indE = p.InitalGapSize
// 	for i := 0; i < p.NOutputs; i++ {
// 		for j := 0; j < p.NHiddens[0]; j++ {
// 			outputWeights[j][i] = outputWeightsTemp[indE]
// 			indE++
// 		}
// 	}

// 	//hidden was row packed with a gap in between
// 	indE = p.InitalGapSize
// 	flag = 0
// 	for i := 0; i < p.NHiddens[0]; i++ {
// 		for j := 0; j < p.NHiddens[1]; j++ {
// 			hiddenWeights[i][j] = hiddenWeightsTemp[indE]
// 			indE++
// 		}
// 		indE += totalGapBetweenHidden
// 		flag++
// 		if flag == (p.NumColsPerCipher) {
// 			indE = indE + totalGapEnd
// 			flag = 0
// 		}
// 	}

// 	/*f4, _ := os.Create("inputWEnc.txt")
// 	f5, _ := os.Create("hiddenWEnc.txt")
// 	f6, _ := os.Create("outputWEnc.txt")
// 	// write a chunk
// 	for i := 0; i < p.NInputs; i++ {
// 		for j := 0; j < p.NHiddens[0]; j++ {
// 			fmt.Fprintf(f4, "%v, ", inputWeights[i][j])
// 		}
// 		fmt.Fprintf(f4, "\n")
// 	}
// 	for i := 0; i < p.NHiddens[0]; i++ {
// 		for j := 0; j < p.NHiddens[1]; j++ {
// 			fmt.Fprintf(f5, "%v, ", hiddenWeights[i][j])
// 		}
// 		fmt.Fprintf(f5, "\n")
// 	}
// 	for i := 0; i < p.NHiddens[1]; i++ {
// 		for j := 0; j < p.NOutputs; j++ {
// 			fmt.Fprintf(f6, "%v, ", outputWeights[i][j])
// 		}
// 		fmt.Fprintf(f6, "\n")
// 	}
// 	f4.Close()
// 	f5.Close()
// 	f6.Close()*/
// 	return [][][]float64{inputWeights, hiddenWeights, outputWeights}
// }
