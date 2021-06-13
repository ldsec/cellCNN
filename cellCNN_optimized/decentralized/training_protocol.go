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
	"runtime"
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
	C               []byte
	W               []byte
	CtC             []byte
	CtW             []byte
}

// ChildUpdatedDataMessage is the message sent by the children to update its local cleartext information
type ChildUpdatedDataMessage struct {
	DC   []byte
	DW   []byte
	CtDC []byte
	CtDW []byte
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

	// Feedback Channels
	FeedbackChannel chan struct{}

	// Other Channels
	AnnouncementChannel chan newIterationAnnouncementStruct
	ChildDataChannel    chan []ChildUpdatedClearDataStruct

	CryptoParams *cellCNN.CryptoParams

	// protocol params
	Path          string
	PartyDataSize int
	XTrain        []*cellCNN.Matrix
	YTrain        []*cellCNN.Matrix

	TrainPlain     bool
	TrainEncrypted bool
	Deterministic  bool
	PrngInt        *cellCNN.PRNGInt

	IterationNumber int
	Epochs          int

	// utils
	Debug bool
}

// NewTrainingProtocol initializes the protocol instance.
func NewTrainingProtocol(n *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
	pap := &TrainingProtocol{
		TreeNodeInstance: n,
		FeedbackChannel:  make(chan struct{}, 1),
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
	if p.Deterministic {

		p.CNNProtocol.C = new(cellCNN.Matrix)
		p.CNNProtocol.C.Rows = cellCNN.Features
		p.CNNProtocol.C.Cols = cellCNN.Filters
		p.CNNProtocol.C.M = make([]complex128, cellCNN.Features*cellCNN.Filters)
		copy(p.CNNProtocol.C.M, cellCNN.C)
		p.CNNProtocol.C.Real = true

		p.CNNProtocol.W = new(cellCNN.Matrix)
		p.CNNProtocol.W.Rows = cellCNN.Filters
		p.CNNProtocol.W.Cols = cellCNN.Classes
		p.CNNProtocol.W.M = make([]complex128, cellCNN.Filters*cellCNN.Classes)
		copy(p.CNNProtocol.W.M, cellCNN.W)
		p.CNNProtocol.W.Real = true
	} else {
		p.CNNProtocol.C = cellCNN.WeightsInit(cellCNN.Features, cellCNN.Filters, cellCNN.Features)
		p.CNNProtocol.W = cellCNN.WeightsInit(cellCNN.Filters, cellCNN.Classes, cellCNN.Filters)
	}

	log.Lvl2("[cellCNN_START]", p.ServerIdentity(), " initialized the weights")

	var bC, bW []byte
	var err error

	// serialize C and W
	if bC, err = p.CNNProtocol.C.MarshalBinary(); err != nil {
		return err
	}

	if bW, err = p.CNNProtocol.W.MarshalBinary(); err != nil {
		return err
	}

	// STEP 2: send clear initial weights down the tree
	if err = p.SendToChildren(&NewIterationMessage{0, bC, bW, nil, nil}); err != nil {
		return fmt.Errorf("send to children: %v", err)
	}

	return nil
}

// Dispatch is called at each node and handle incoming messages.
func (p *TrainingProtocol) Dispatch() error {

	defer p.Done()

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
				if err = CMatrix.UnmarshalBinary(msg.C); err != nil {
					return err
				}

				if err = WMatrix.UnmarshalBinary(msg.W); err != nil {
					return err
				}

				p.CNNProtocol.SetWeights(CMatrix, WMatrix)
				if p.TrainEncrypted {
					p.CNNProtocol.EncryptWeights()
				}

				log.Lvl2("[cellCNN_START]", p.ServerIdentity(), " initialized the weights")

			} else {

				// replace weights
				if p.TrainPlain {

					if err = p.CNNProtocol.C.UnmarshalBinary(msg.C); err != nil {
						return err
					}

					if err = p.CNNProtocol.W.UnmarshalBinary(msg.W); err != nil {
						return err
					}
				}

				if p.TrainEncrypted {

					if err = p.CNNProtocol.CtC.UnmarshalBinary(msg.CtC); err != nil {
						return err
					}

					if err = p.CNNProtocol.CtW.UnmarshalBinary(msg.CtW); err != nil {
						return err
					}
				}
			}

			//need to check as new number is part of the message for non root nodes
			finished = p.IterationNumber >= p.Epochs

		} else if p.IterationNumber == 0 && p.IsRoot() {

			if p.TrainEncrypted {
				p.CNNProtocol.EncryptWeights()
			}
		}

		if !finished {

			// get results of local iterations
			if err := p.ascendingUpdateGeneralModelPhase(); err != nil {
				return err
			}

			if p.IsRoot() {

				p.IterationNumber = p.IterationNumber + 1

				var bC, bW, bCtC, bCtW []byte
				var err error

				// STEP 4. update weights and broadcast the updated weights
				if p.TrainPlain {

					p.CNNProtocol.UpdatePlain()

					if bC, err = p.CNNProtocol.C.MarshalBinary(); err != nil {
						return err
					}

					if bW, err = p.CNNProtocol.W.MarshalBinary(); err != nil {
						return err
					}
				}

				if p.TrainEncrypted {

					p.CNNProtocol.Update()

					if bCtC, err = p.CNNProtocol.CtC.MarshalBinary(); err != nil {
						return err
					}

					if bCtW, err = p.CNNProtocol.CtW.MarshalBinary(); err != nil {
						return err
					}
				}

				// send new weights
				if err := p.SendToChildren(&NewIterationMessage{p.IterationNumber, bC, bW, bCtC, bCtW}); err != nil {
					return fmt.Errorf("send to children: %v", err)
				}
			}
		}
	}

	// STEP 4. Report final weights
	if p.IsRoot() {

		log.Lvl2("Loading Validation Data...")
		XValid, YValid := cellCNN.LoadValidDataFrom("../../normalized/", 2000, cellCNN.Cells, cellCNN.Features)
		log.Lvl2("Done")

		if p.TrainPlain && p.TrainEncrypted {
			p.CNNProtocol.PrintCtCPrecision(p.CryptoParams.AggregateSk)
			p.CNNProtocol.PrintCtWPrecision(p.CryptoParams.AggregateSk)
		} else if p.TrainPlain {
			p.CNNProtocol.C.Print()
			p.CNNProtocol.W.Print()
		} else if p.TrainEncrypted {

			cellCNN.DecryptPrint(cellCNN.Features, cellCNN.Filters, true, p.CNNProtocol.CtDC, *p.CryptoParams.Params, p.CryptoParams.AggregateSk)

			for i := 0; i < cellCNN.Classes; i++ {
				cellCNN.DecryptPrint(1, cellCNN.Filters, true, p.CNNProtocol.Eval.RotateNew(p.CNNProtocol.CtDW, i*cellCNN.BatchSize*cellCNN.Filters), *p.CryptoParams.Params, p.CryptoParams.AggregateSk)
			}
		}

		r := 0
		for i := 0; i < 2000/cellCNN.BatchSize; i++ {

			XPrePool := new(cellCNN.Matrix)
			XBatch := cellCNN.NewMatrix(cellCNN.BatchSize, cellCNN.Features)
			YBatch := cellCNN.NewMatrix(cellCNN.BatchSize, cellCNN.Classes)

			for j := 0; j < cellCNN.BatchSize; j++ {

				X := XValid[cellCNN.BatchSize*i+j]
				Y := YValid[cellCNN.BatchSize*i+j]

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
				precisionStats := ckks.GetPrecisionStats(*p.CryptoParams.Params, p.CNNProtocol.Encoder, nil, v.M, ctv.M, p.CryptoParams.Params.LogSlots(), 0)
				fmt.Printf("Batch[%2d]", i)
				fmt.Println(precisionStats.String())
			}

			var y int
			for i := 0; i < cellCNN.BatchSize; i++ {

				if real(v.M[i*2]) > real(v.M[i*2+1]) {
					y = 1
				} else {
					y = 0
				}

				if y != int(real(YBatch.M[i*2])) {
					r++
				}
			}
		}

		log.Lvl2("error :", 100.0*float64(r)/float64(2000), "%")

		time.Sleep(10 * time.Second)
		p.FeedbackChannel <- struct{}{}
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

	runtime.GC() // Forces garbage collection

	return nil
}

func (p *TrainingProtocol) localComputation() {

	XPrePool := new(cellCNN.Matrix)
	XBatch := cellCNN.NewMatrix(cellCNN.BatchSize, cellCNN.Features)
	YBatch := cellCNN.NewMatrix(cellCNN.BatchSize, cellCNN.Classes)

	// Pre-pools the cells
	for k := 0; k < cellCNN.BatchSize; k++ {

		randi := p.PrngInt.RandInt()

		X := p.XTrain[randi]
		Y := p.YTrain[randi]

		XPrePool.SumColumns(X)
		XPrePool.MultConst(XPrePool, complex(1.0/float64(cellCNN.Cells), 0))

		XBatch.SetRow(k, XPrePool.M)
		YBatch.SetRow(k, []complex128{Y.M[1], Y.M[0]})
	}

	if p.TrainPlain {
		p.CNNProtocol.ForwardPlain(XBatch)
		p.CNNProtocol.BackWardPlain(XBatch, YBatch, p.Tree().Size()) // takes care of pre-applying 1/#Parties
	}

	start := time.Now()

	if p.TrainEncrypted {
		p.CNNProtocol.Forward(XBatch)
		p.CNNProtocol.Refresh(p.CryptoParams.AggregateSk, p.Tree().Size())
		p.CNNProtocol.Backward(XBatch, YBatch, p.Tree().Size())
	}

	log.Lvlf2("Iter[%d] : %s", p.IterationNumber, time.Since(start))
}

func (p *TrainingProtocol) broadcast() error {

	if !p.IsLeaf() {

		// This reads all the data that were sent by all the children
		for _, v := range <-p.ChildDataChannel {

			// Deserialize child contribution and aggregates on local weights

			if p.TrainPlain {

				childDC := new(cellCNN.Matrix)
				childDW := new(cellCNN.Matrix)

				if err := childDC.UnmarshalBinary(v.DC); err != nil {
					return err
				}

				if err := childDW.UnmarshalBinary(v.DW); err != nil {
					return err
				}

				p.CNNProtocol.DC.Add(p.CNNProtocol.DC, childDC)
				p.CNNProtocol.DW.Add(p.CNNProtocol.DW, childDW)

			}

			if p.TrainEncrypted {

				childDC := p.CNNProtocol.CtDC.CopyNew()
				childDW := p.CNNProtocol.CtDW.CopyNew()

				if err := childDC.UnmarshalBinary(v.CtDC); err != nil {
					return err
				}

				if err := childDW.UnmarshalBinary(v.CtDW); err != nil {
					return err
				}

				p.CNNProtocol.Eval.Add(p.CNNProtocol.CtDC, childDC, p.CNNProtocol.CtDC)
				p.CNNProtocol.Eval.Add(p.CNNProtocol.CtDW, childDW, p.CNNProtocol.CtDW)
			}
		}
	}

	// If leaf or not root, send DWPool and DCPool up the tree
	if !p.IsRoot() {

		// Serialize local weights

		var bDC, bDW, bCtDC, bCtDW []byte
		var err error

		if p.TrainPlain {

			if bDC, err = p.CNNProtocol.DC.MarshalBinary(); err != nil {
				return err
			}

			if bDW, err = p.CNNProtocol.DW.MarshalBinary(); err != nil {
				return err
			}
		}

		if p.TrainEncrypted {

			if bCtDC, err = p.CNNProtocol.CtDC.MarshalBinary(); err != nil {
				return err
			}

			if bCtDW, err = p.CNNProtocol.CtDW.MarshalBinary(); err != nil {
				return err
			}
		}

		if err = p.SendToParent(&ChildUpdatedDataMessage{
			DC:   bDC,
			DW:   bDW,
			CtDC: bCtDC,
			CtDW: bCtDW,
		}); err != nil {
			return fmt.Errorf("send to parent: %v", err)
		}
	}

	return nil
}
