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
func NewBootstrapProtocol(n *onet.TreeNodeInstance, cryptoParams *cellCNN.CryptoParams, seed []byte) (*BootstrapProtocol, error) {
	p := &BootstrapProtocol{
		TreeNodeInstance: n,
		FeedbackChannel:  make(chan *ckks.Ciphertext, 1),
		rootInited:       make(chan struct{}),
	}

	if err := p.RegisterChannels(&p.SyncChannel, &p.AnnouncementBootChannel, &p.BootSharesChannel); err != nil {
		return nil, fmt.Errorf("couldn't register channel: %v", err)
	}

	if err := p.init(cryptoParams, seed); err != nil {
		return nil, fmt.Errorf("init protocol: %v", err)
	}

	return p, nil
}

// NewBootstrapProtocolFunction returns a function generating a new bootstrap protocol instance properly initialized.
// It is intended for use with onet.
func NewBootstrapProtocolFunction(cryptoParams *cellCNN.CryptoParams, seed []byte) func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
	return func(tni *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {
		return NewBootstrapProtocol(tni, cryptoParams, seed)
	}
}

// init setups the protocol, think of it as a constructor but working with Onet.
func (p *BootstrapProtocol) init(cp *cellCNN.CryptoParams, randomBytes []byte) error {
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

	/*var err error
	prng, err := utils.NewKeyedPRNG(randomBytes)
	if err != nil {
		return fmt.Errorf("creating a new PRNG: %v", err)
	}

	ringQP, err := ring.NewRing(p.cryptoParams.Params.N(), append(p.cryptoParams.Params.Qi(), p.cryptoParams.Params.Pi()...))
	if err != nil {
		return fmt.Errorf("creating new ring: %v", err)
	}
	p.sampler = ring.NewUniformSampler(prng, ringQP)*/

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

	/*if err := p.createShares(); err != nil {
		return fmt.Errorf("create shares: %v", err)
	}*/

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
			/*shares := <-p.BootSharesChannel

			rSharesDecryptNew, rSharesRecryptNew, _ := p.convertBootSharesReceive(shares.RSharesDecryptBytes, shares.RSharesRecryptBytes)

			mutex.Lock()
			for i := range shares.RSharesDecryptBytes {
				p.refreshProto.Aggregate(p.rSharesDecrypt[i], rSharesDecryptNew[i], p.rSharesDecrypt[i])
				p.refreshProto.Aggregate(p.rSharesRecrypt[i], rSharesRecryptNew[i], p.rSharesRecrypt[i])
			}
			mutex.Unlock()*/
		}(childIdx)
	}
	wg.Wait()

	//check if its the root then aggregate else wait on the parent.
	if p.IsRoot() {
		/*for i, ct := range p.dataToBootstrap {
			p.refreshProto.Decrypt(ct, p.rSharesDecrypt[i])
			p.refreshProto.Recode(ct)
			p.refreshProto.Recrypt(ct, p.listCRS[i], p.rSharesRecrypt[i])
			p.dataToBootstrap = ct
		}*/
		log.Lvl2("time: ", time.Since(protocolTime))
		p.FeedbackChannel <- p.dataToBootstrap
	} else {
		//send the shares to the parent...
		log.Lvl2("Sending my PCKSShare")
		// marshal to binary
		/*rSharesDecryptBytes, rSharesRecryptBytes, err := p.convertBootSharesToSend()
		if err != nil {
			return err
		}*/

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

	// unmarshal and create shares
	/*dtb := ckks.NewCiphertext(p.cryptoParams.Params, )
	err := dtb.UnmarshalBinary(p.cryptoParams, msg.DataToBootstrapBytes)
	if err != nil {
		return err
	}
	p.dataToBootstrap = dtb

	err = p.createShares()
	if err != nil {
		return err
	}*/

	return nil
}

/*func (p *BootstrapProtocol) createShares() error {
	if err := p.ensureIsInit(); err != nil {
		return fmt.Errorf("check init'ed: %v", err)
	}

	//Parameters for refresh
	p.refreshProto = dckks.NewRefreshProtocol(*p.cryptoParams.Params)

	shareDec, shareRec := p.refreshProto.AllocateShares(p.dataToBootstrap.Level())
	p.rSharesDecrypt = shareDec
	p.rSharesRecrypt = shareRec
	p.listCRS = p.sampler.ReadNew()
	p.refreshProto.GenShares(p.cryptoParams.Sk.Value, p.dataToBootstrap.Level(), len(p.Roster().List), p.dataToBootstrap, p.cryptoParams.Params.Scale(), p.listCRS, shareDec, shareRec)

	return nil
}

func (p *BootstrapProtocol) convertBootSharesToSend() ([]byte, []byte, error) {

	rshareDecPoly := ring.Poly{Coeffs: (*p.rSharesDecrypt).Coeffs}
	rSharesDecryptBytes, err := (&rshareDecPoly).MarshalBinary()
	if err != nil {
		return nil, nil, err
	}

	rshareRecPoly := ring.Poly{Coeffs: (*p.rSharesRecrypt).Coeffs}
	rSharesRecryptBytes, err := (&rshareRecPoly).MarshalBinary()
	if err != nil {
		return nil, nil, err
	}


	return rSharesDecryptBytes, rSharesRecryptBytes, nil
}

func (p *BootstrapProtocol) convertBootSharesReceive(rSharesDecryptBytes []byte, rSharesRecryptBytes []byte) (dckks.RefreshShareDecrypt, dckks.RefreshShareRecrypt, error) {

	tmpShareDec, tmpShareRec := p.refreshProto.AllocateShares(p.dataToBootstrap.Level())

	rshareDecPoly := ring.Poly{Coeffs: (tmpShareDec).Coeffs}
	err := (&rshareDecPoly).UnmarshalBinary(rSharesDecryptBytes)
	if err != nil {
		return nil, nil, err
	}
	tmpShareDec.Coeffs = rshareDecPoly.Coeffs
	rSharesDecrypt := tmpShareDec

	rshareRecPoly := ring.Poly{Coeffs: (tmpShareRec).Coeffs}
	err = (&rshareRecPoly).UnmarshalBinary(rSharesRecryptBytes)
	if err != nil {
		return nil, nil, err
	}
	tmpShareRec.Coeffs = rshareRecPoly.Coeffs
	rSharesRecrypt := tmpShareRec

	return rSharesDecrypt, rSharesRecrypt, nil
}*/
