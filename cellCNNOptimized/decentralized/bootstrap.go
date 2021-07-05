package decentralized

import (
	"errors"
	"fmt"
	"github.com/ldsec/cellCNN/cellCNNOptimized"
	"sync"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/ring"
	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"
	"go.dedis.ch/onet/v3/network"
)

// BootstrapProtocolName is the registered name for the bootstrap protocol
const BootstrapProtocolName = "BootstrapProtocol"

func init() {
	network.RegisterMessages(
		new(SyncMessage),
		new(AnnouncementBootMessage),
		new(BootSharesMessage),
	)
}

// Messages
//______________________________________________________________________________________________________________________

// BootSharesMessage is the message sent by the children with the shares to perform a bootstrap
type BootSharesMessage struct {
	RSharesDecryptBytes []byte
	RSharesRecryptBytes []byte
}

// AnnouncementBootMessage is the message sent by to the children with the data to key switch and key
type AnnouncementBootMessage struct {
	DataToBootstrapBytes []byte
}

// Structs
//______________________________________________________________________________________________________________________
type announcementBootStruct struct {
	*onet.TreeNode
	AnnouncementBootMessage
}

type bootSharesStruct struct {
	*onet.TreeNode
	BootSharesMessage
}

// BootstrapProtocol struct stores the context for onet
type BootstrapProtocol struct {
	*onet.TreeNodeInstance

	Sync bool

	// Protocol feedback channel
	FeedbackChannel chan *ckks.Ciphertext

	// Protocol communication channels
	AnnouncementBootChannel chan announcementBootStruct
	BootSharesChannel       chan bootSharesStruct
	SyncChannel         	chan SyncStruct

	//Params ckks parameters.
	cryptoParams *cellCNN.CryptoParams

	dataToBootstrap *ckks.Ciphertext

	rootInited chan struct{}

	sampler *ring.UniformSampler
	listCRS *ring.Poly

	refreshProto   *dckks.RefreshProtocol
	rSharesDecrypt dckks.RefreshShareDecrypt
	rSharesRecrypt dckks.RefreshShareRecrypt
}

// NewBootstrapProtocol instantiates and initializes a new bootstrap protocol instance.
func NewBootstrapProtocol(n *onet.TreeNodeInstance, cryptoParams *cellCNN.CryptoParams) (*BootstrapProtocol, error) {
	p := &BootstrapProtocol{
		TreeNodeInstance: n,
		FeedbackChannel:  make(chan *ckks.Ciphertext, 1),
		rootInited:       make(chan struct{}),
	}

	if err := p.RegisterChannels(&p.SyncChannel, &p.AnnouncementBootChannel, &p.BootSharesChannel); err != nil {
		return nil, fmt.Errorf("couldn't register channel: %v", err)
	}

	if err := p.init(cryptoParams); err != nil {
		return nil, fmt.Errorf("init protocol: %v", err)
	}

	return p, nil
}

// NewBootstrapProtocolFunction returns a function generating a new bootstrap protocol instance properly initialized.
// It is intended for use with onet.
func NewBootstrapProtocolFunction(cryptoParams *cellCNN.CryptoParams) func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
	return func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
		return NewBootstrapProtocol(tni, cryptoParams)
	}
}

// init setups the protocol, think of it as a constructor but working with Onet.
func (p *BootstrapProtocol) init(cp *cellCNN.CryptoParams) error {
	if cp == nil {
		return errors.New("CryptoParams are nil")
	}

	if err := p.ensureIsInit(); err == nil {
		return errors.New("already init'ed")
	}

	p.cryptoParams = cp
	p.Sync = true

	if p.IsRoot() {
		p.dataToBootstrap = ckks.NewCiphertext(*p.cryptoParams.Params, 1, p.cryptoParams.Params.MaxLevel(), p.cryptoParams.Params.Scale())
	}

	return nil
}

func (p *BootstrapProtocol) ensureIsInit() error {
	if p.cryptoParams == nil {
		return errors.New("no crypto params set")
	}

	return nil
}

// InitRoot initializes the variables for the protocol. Should be called before the dispatch
func (p *BootstrapProtocol) InitRoot(ct *ckks.Ciphertext) error {
	if err := p.ensureIsInit(); err != nil {
		return fmt.Errorf("not init'ed: %v", err)
	}

	p.dataToBootstrap = ct

	close(p.rootInited)

	return nil
}

// Start starts the protocol only at root
func (p *BootstrapProtocol) Start() error {
	log.Lvl2(p.ServerIdentity(), " starting a collective bootstrap")

	dtbb, err := p.dataToBootstrap.MarshalBinary()
	if err != nil {
		return err
	}

	log.Lvl2("Send Data (bytes):", len(dtbb)/2)
	if err := p.SendToChildrenInParallel(&AnnouncementBootMessage{
		DataToBootstrapBytes: dtbb[:len(dtbb)/2],
	}); err != nil {
		return fmt.Errorf("could not send start message: %v", err)
	}

	return nil
}

// Dispatch is called at each node to then run the protocol
func (p *BootstrapProtocol) Dispatch() error {
	defer p.Done()

	if p.Sync {
		err := SyncProtocol(p.TreeNodeInstance, p.SyncChannel)
		if err != nil {
			return err
		}
		p.Sync = false
	}

	var protocolTime time.Time
	if p.IsRoot() {
		protocolTime = time.Now()
	}

	if !p.IsRoot() {
		err := p.announcementPhase()
		if err != nil {
			return err
		}
	}

	var wg sync.WaitGroup
	//mutex := sync.Mutex{}
	for childIdx := range p.Children() {
		wg.Add(1)
		go func(c int) {
			defer wg.Done()
			log.Lvl2("Getting a child PCKSShares")
			<-p.BootSharesChannel
		}(childIdx)
	}
	wg.Wait()

	//check if its the root then aggregate else wait on the parent.
	if p.IsRoot() {
		log.Lvl2("time: ", time.Since(protocolTime))
		p.FeedbackChannel <- p.dataToBootstrap
	} else {
		//send the shares to the parent...
		log.Lvl2("Sending my PCKSShare")
		ct := ckks.NewCiphertext(*p.cryptoParams.Params, 1, p.cryptoParams.Params.MaxLevel(), p.cryptoParams.Params.Scale())
		ctb, err := ct.MarshalBinary()
		if err != nil {
			return err
		}

		log.Lvl2("Send Data (bytes):", len(ctb))
		err = p.SendToParent(&BootSharesMessage{RSharesDecryptBytes: ctb})
		if err != nil {
			return err
		}
	}

	return nil
}

// Announce forwarding down the tree.
func (p *BootstrapProtocol) announcementPhase() error {
	// wait for the message from the root to start the protocol
	msg := <-p.AnnouncementBootChannel
	// if it is not a leaf node it propagates the message to the its children (in a tree-like way)
	if !p.IsLeaf() {
		log.Lvl2("Send Data (bytes):", len(msg.DataToBootstrapBytes))
		if err := p.SendToChildrenInParallel(&AnnouncementBootMessage{
			DataToBootstrapBytes: msg.DataToBootstrapBytes,
		}); err != nil {
			return fmt.Errorf("error sending <AnnouncementBootMessage>: %v", err)
		}
	}

	return nil
}