package decentralized

import "C"
import (
	"errors"
	"fmt"
	"github.com/ldsec/cellCNN/cellCNN_optimized"
	"github.com/ldsec/lattigo/v2/ckks"
	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"
	"go.dedis.ch/onet/v3/network"
	"math/rand"
	"time"
)

func init() {
	network.RegisterMessages(
		new(NewIterationMessage),
		new(ChildUpdatedDataMessage),
	)
}

// TrainingProtocolName is the registered name for the regression_encrypted training protocol.
const TrainingProtocolName = "TrainingProtocol"

// Messages
//______________________________________________________________________________________________________________________

// NewIterationMessage is the message sent by the root node to start a new global iteration
type NewIterationMessage struct {
	IterationNumber int
	C 	[]byte
	W 	[]byte
	CtC []byte
	CtW []byte
}

// ChildUpdatedDataMessage is the message sent by the children to update its local cleartext information
type ChildUpdatedDataMessage struct {
	DC []byte
	DW []byte
}

// Structs
//______________________________________________________________________________________________________________________
type newIterationAnnouncementStruct struct {
	*onet.TreeNode
	NewIterationMessage
}

// ChildUpdatedClearDataStruct is the message sent by the children to update its local cleartext information
type ChildUpdatedClearDataStruct struct {
	*onet.TreeNode
	ChildUpdatedDataMessage
}

// TrainingProtocol keeps the state of the protocol
type TrainingProtocol struct {
	*onet.TreeNodeInstance

	CNNProtocol *cellCNN.CellCNNProtocol

	// Root Channel
	WaitChannel chan struct{}

	// Feedback Channels
	FeedbackChannel chan struct{}

	// Other Channels
	AnnouncementChannel chan newIterationAnnouncementStruct
	ChildDataChannel    chan []ChildUpdatedClearDataStruct

	CryptoParams *cellCNN.CryptoParams

	// protocol params
	Path			string
	PartyDataSize   int
	XTrain 			[]*cellCNN.Matrix
	YTrain			[]*cellCNN.Matrix

	TrainEncrypted  bool

	IterationNumber int
	Epochs          int

	Samples 		int
	Cells			int
	Features		int
	Filters			int
	Classes			int

	DWPool 			*cellCNN.Matrix
	DCPool			*cellCNN.Matrix

	ctDWPool		*ckks.Ciphertext
	ctDCPool 		*ckks.Ciphertext

	// utils
	Debug bool
}

// NewTrainingProtocol initializes the protocol instance.
func NewTrainingProtocol(n *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
	pap := &TrainingProtocol{
		TreeNodeInstance:    n,
		WaitChannel:         make(chan struct{}, 1),
		FeedbackChannel:     make(chan struct{}, 1),
	}

	err := pap.RegisterChannels(&pap.AnnouncementChannel, &pap.ChildDataChannel)
	if err != nil {
		return nil, errors.New("couldn't register announcement channel: " + err.Error())
	}

	return pap, nil
}

// Start is called at the root to begin the execution of the protocol.
func (p *TrainingProtocol) Start() error {
	log.Lvl2("[cellCNN_START]", p.ServerIdentity(), " started a cellCNN Protocol")

	// STEP 1: weight init
	CMatrix := cellCNN.WeightsInit(p.Features, p.Filters, p.Features)
	WMatrix := cellCNN.WeightsInit(p.Filters, p.Classes, p.Filters)

	log.Lvl2("[cellCNN_START]", p.ServerIdentity(), " initialized the weights")

	// serialize C and W
	Cb, err := CMatrix.MarshalBinary()
	if err != nil {
		return err
	}
	Wb, err := WMatrix.MarshalBinary()
	if err != nil {
		return err
	}

	// STEP 2: send clear initial weights down the tree
	newIterationMessage := NewIterationMessage{0, Cb, Wb, nil, nil}
	if err := p.SendToChildren(&newIterationMessage); err != nil {
		return fmt.Errorf("send to children: %v", err)
	}

	// STEP 3. set and encrypt weights
	p.CNNProtocol.SetWeights(CMatrix, WMatrix)
	if p.TrainEncrypted {
		p.CNNProtocol.EncryptWeights()
	}

	// Unlock channel (root node can continue with Dispatch)
	p.WaitChannel <- struct{}{}
	close(p.WaitChannel)

	return nil
}

// Dispatch is called at each node and handle incoming messages.
func (p *TrainingProtocol) Dispatch() error {
	defer p.Done()

	// Wait for the initialization of the weights (this is done in the Start())
	if p.IsRoot() {
		<-p.WaitChannel
	}

	for p.IterationNumber < p.Epochs { // protocol iterations
		// 1. Forward Pass announcement phase
		finished := false
		if !p.IsRoot() {
			// if it is not the root each node gets and forwards the general updated weights down
			msg, err := p.newIterationAnnouncementPhase()
			if err != nil {
				return fmt.Errorf("encrypted iteration announcement phase: %v", err)
			}
			p.IterationNumber = msg.IterationNumber

			if p.IterationNumber == 0 {
				// STEP 1: weight init
				CMatrix := new(cellCNN.Matrix)
				WMatrix := new(cellCNN.Matrix)

				// STEP 3. set and encrypt weights
				err = CMatrix.UnmarshalBinary(msg.C)
				if err != nil {
					return err
				}
				err = WMatrix.UnmarshalBinary(msg.W)
				if err != nil {
					return err
				}

				p.CNNProtocol.SetWeights(CMatrix, WMatrix)
				if p.TrainEncrypted {
					p.CNNProtocol.EncryptWeights()
				}

				log.Lvl2("[cellCNN_START]", p.ServerIdentity(), " initialized the weights")

			} else {
				// replace weights
				if !p.TrainEncrypted {
					err = p.CNNProtocol.C.UnmarshalBinary(msg.C)
					if err != nil {
						return err
					}
					err = p.CNNProtocol.W.UnmarshalBinary(msg.W)
					if err != nil {
						return err
					}
				} else {
					err = p.CNNProtocol.CtC().UnmarshalBinary(msg.CtC)
					if err != nil {
						return err
					}
					err = p.CNNProtocol.CtW().UnmarshalBinary(msg.CtW)
					if err != nil {
						return err
					}
				}
			}

			//need to check as new number is part of the message for non root nodes
			finished = p.IterationNumber >= p.Epochs
		}

		if !finished {
			// get results of local iterations
			err := p.ascendingUpdateGeneralModelPhase()
			if err != nil {
				return err
			}

			if p.IsRoot() {
				p.IterationNumber = p.IterationNumber + 1

				var newIterationMessage NewIterationMessage
				// STEP 4. update weights
				if !p.TrainEncrypted {

					p.CNNProtocol.UpdatePlain(p.DCPool, p.DWPool)

					// serialize C and W
					Cb, err := p.CNNProtocol.C.MarshalBinary()
					if err != nil {
						return err
					}
					Wb, err := p.CNNProtocol.W.MarshalBinary()
					if err != nil {
						return err
					}

					newIterationMessage = NewIterationMessage{p.IterationNumber, Cb, Wb, nil, nil}
				} else {
					p.CNNProtocol.Update(p.ctDCPool, p.ctDWPool)
					bCtC, err := p.CNNProtocol.CtC().MarshalBinary()
					if err != nil {
						return err
					}
					bCtW, err := p.CNNProtocol.CtW().MarshalBinary()
					if err != nil {
						return err
					}
					newIterationMessage = NewIterationMessage{p.IterationNumber, nil, nil, bCtC, bCtW}
				}
				// send new weights
				if err := p.SendToChildren(&newIterationMessage); err != nil {
					return fmt.Errorf("send to children: %v", err)
				}
			}
		}
	}

	// STEP 4. Report final weights
	if p.IsRoot() {
		time.Sleep(10*time.Second)
		p.FeedbackChannel <- struct{}{}

		log.Lvl2("Loading Validation Data...")
		XValid, YValid := cellCNN.LoadValidDataFrom("../../normalized/", 2000, cellCNN.Cells, cellCNN.Features)
		log.Lvl2("Done")
		p.CNNProtocol.C.Print()
		p.CNNProtocol.W.Print()

		r := 0
		for i := 0; i < 2000/cellCNN.BatchSize; i++{

			XPrePool := new(cellCNN.Matrix)
			XBatch := cellCNN.NewMatrix(cellCNN.BatchSize, cellCNN.Features)
			YBatch := cellCNN.NewMatrix(cellCNN.BatchSize, cellCNN.Classes)

			for j := 0; j < cellCNN.BatchSize; j++ {

				X := XValid[cellCNN.BatchSize * i + j]
				Y := YValid[cellCNN.BatchSize * i + j]

				XPrePool.SumColumns(X)
				XPrePool.MultConst(XPrePool, complex(1.0/float64(cellCNN.Cells), 0))

				XBatch.SetRow(j, XPrePool.M)
				YBatch.SetRow(j, []complex128{Y.M[1], Y.M[0]})
			}

			v := p.CNNProtocol.PredictPlain(XBatch)

			if p.TrainEncrypted {
				v.Print()
				ctv := p.CNNProtocol.Predict(XBatch, p.CryptoParams.AggregateSk)
				ctv.Print()
				precisionStats := ckks.GetPrecisionStats(*p.CryptoParams.Params, p.CNNProtocol.Encoder(), nil, v.M, ctv.M, p.CryptoParams.Params.LogSlots(), 0)
				fmt.Printf("Batch[%2d]", i)
				fmt.Println(precisionStats.String())
			}
			
			var y int
			for i := 0; i < cellCNN.BatchSize; i++{

				if real(v.M[i*2]) > real(v.M[i*2+1]){
					y = 1
				}else{
					y = 0
				}

				if y != int(real(YBatch.M[i*2])){
					r++
				}
			}
		}

		log.Lvl2("error :", 100.0*float64(r)/float64(2000), "%")
	}

	return nil
}

// Announce forwarding down the tree.
func (p *TrainingProtocol) newIterationAnnouncementPhase() (NewIterationMessage, error) {
	// wait for the message from the root to start the protocol
	newEncryptedIterationMessage := <-p.AnnouncementChannel

	// if it is not a leaf node it propagates the message to the its children (in a tree-like way)
	if !p.IsLeaf() {
		if err := p.SendToChildren(&newEncryptedIterationMessage.NewIterationMessage); err != nil {
			return NewIterationMessage{}, fmt.Errorf("send to children: %v", err)
		}

	}
	return newEncryptedIterationMessage.NewIterationMessage, nil
}

// Local updates, Results pushing up the tree containing aggregation results.
func (p *TrainingProtocol) ascendingUpdateGeneralModelPhase() error {

	p.localComputation()

	err := p.broadcast()
	if err != nil {
		return err
	}

	return nil
}

func (p *TrainingProtocol) LoadTrainingData() ([]*cellCNN.Matrix, []*cellCNN.Matrix){

	log.Lvl2("Loading Training Data ...")
	XTrain, YTrain := cellCNN.LoadTrainDataFrom(p.Path, p.Samples, p.Cells, p.Features)
	log.Lvl2("Done")

	return XTrain, YTrain
}

func (p *TrainingProtocol) localComputation() {

	XPrePool := new(cellCNN.Matrix)
	XBatch := cellCNN.NewMatrix(cellCNN.BatchSize, cellCNN.Features)
	YBatch := cellCNN.NewMatrix(cellCNN.BatchSize, cellCNN.Classes)

	// Pre-pools the cells
	for k := 0; k < cellCNN.BatchSize; k++ {

		randi := rand.Intn(p.PartyDataSize)

		X := p.XTrain[randi]
		Y := p.YTrain[randi]

		XPrePool.SumColumns(X)
		XPrePool.MultConst(XPrePool, complex(1.0/float64(cellCNN.Cells), 0))

		XBatch.SetRow(k, XPrePool.M)
		YBatch.SetRow(k, []complex128{Y.M[1], Y.M[0]})
	}

	start := time.Now()
	// === Cleartext ===
	if !p.TrainEncrypted {
		p.CNNProtocol.ForwardPlain(XBatch)
		p.CNNProtocol.BackWardPlain(XBatch, YBatch, p.Tree().Size()) // takes care of pre-applying 1/#Parties
	} else { // === Ciphertext ===
		p.CNNProtocol.Forward(XBatch)
		newCtBoot := p.CNNProtocol.Refresh(p.CryptoParams.AggregateSk, p.CNNProtocol.CtBoot(), p.Tree().Size())
		p.CNNProtocol.SetCtBoot(newCtBoot)
		p.CNNProtocol.Backward(XBatch, YBatch, p.Tree().Size())

		/*
			log.Lvl2("DC")
			p.CNNProtocol.DC.Print()
			cellCNN.DecryptPrint(cellCNN.Features, cellCNN.Filters, true, p.CNNProtocol.CtDC(), *p.CryptoParams.Params, p.CryptoParams.AggregateSk)

			log.Lvl2("DW")
			p.CNNProtocol.DW.Transpose().Print()
			for i := 0; i < cellCNN.Classes; i++{
				cellCNN.DecryptPrint(1, cellCNN.Filters, true, p.CNNProtocol.Eval().RotateNew(p.CNNProtocol.CtDW(), i*cellCNN.BatchSize*cellCNN.Filters), *p.CryptoParams.Params, p.CryptoParams.AggregateSk)
			}
		*/
	}
	log.Lvlf2("Iter[%d] : %s", p.IterationNumber, time.Since(start))
}

func (p *TrainingProtocol) broadcast() error {

	if !p.TrainEncrypted {
		p.DWPool = cellCNN.NewMatrix(cellCNN.Filters, cellCNN.Classes)
		p.DCPool = cellCNN.NewMatrix(cellCNN.Features, cellCNN.Filters)
	} else {
		p.ctDWPool = p.CNNProtocol.CtDW().CopyNew()
		p.ctDCPool = p.CNNProtocol.CtDC().CopyNew()
	}

	// If leaf, directly aggregates on DC and DW Pool (else it sends 0 values)
	if p.IsLeaf(){
		p.DCPool.Add(p.DCPool, p.CNNProtocol.DC)
		p.DWPool.Add(p.DWPool, p.CNNProtocol.DW)
	}

	// If not leaf, waits on the children and 
	// aggregates from the children in the DC and DW pools
	if !p.IsLeaf() {
		// this reads all the data that were sent by all the children
		for _, v := range <-p.ChildDataChannel {

			// deserialize child contribution
			if !p.TrainEncrypted{
				childDC := new(cellCNN.Matrix)
				err := childDC.UnmarshalBinary(v.DC)
				if err != nil {
					return err
				}
				childDW := new(cellCNN.Matrix)
				err = childDW.UnmarshalBinary(v.DW)
				if err != nil {
					return err
				}

				p.DCPool.Add(p.DCPool, childDC)
				p.DWPool.Add(p.DWPool, childDW)

			} else {
				childDW := p.CNNProtocol.CtDW().CopyNew()
				childDC := p.CNNProtocol.CtDC().CopyNew()

				err := childDW.UnmarshalBinary(v.DW)
				if err != nil {
					return err
				}
				err = childDC.UnmarshalBinary(v.DC)
				if err != nil {
					return err
				}

				p.CNNProtocol.Eval().Add(p.ctDWPool, childDW, p.ctDWPool)
				p.CNNProtocol.Eval().Add(p.ctDCPool, childDC, p.ctDCPool)
			}
		}
	}

	// If leaf or not root, send DWPool and DCPool up the tree
	if !p.IsRoot() {
		// serialize DCPool and DWPool
		if !p.TrainEncrypted {
			DCb, err := p.DCPool.MarshalBinary()
			if err != nil {
				return err
			}
			DWb, err := p.DWPool.MarshalBinary()
			if err != nil {
				return err
			}

			err = p.SendToParent(&ChildUpdatedDataMessage{
				DC: DCb,
				DW: DWb,
			})
			if err != nil {
				return fmt.Errorf("send to parent: %v", err)
			}
		} else {
			bDW, err := p.ctDWPool.MarshalBinary()
			if err != nil {
				return err
			}
			bDC, err := p.ctDCPool.MarshalBinary()
			if err != nil {
				return err
			}

			err = p.SendToParent(&ChildUpdatedDataMessage{
				DC: bDW,
				DW: bDC,
			})
			if err != nil {
				return fmt.Errorf("send to parent: %v", err)
			}
		}
	}

	if p.Tree().Size() == 1 {
		p.DCPool.Add(p.DCPool, p.CNNProtocol.DC)
		p.DWPool.Add(p.DWPool, p.CNNProtocol.DW)
	}

	return nil
}

