package cellCNN

import(
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/drlwe"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
)


// =========================
// Encryption Key Generation
// =========================

func (c *CellCNNProtocol) NewCKGProtocol(){
	c.ckgProtocol = dckks.NewCKGProtocol(c.params)
	c.ckgShare = c.ckgProtocol.AllocateShares()
}

func (c *CellCNNProtocol) CKGGenShare(crp *ring.Poly){
	c.ckgProtocol.GenShare(&c.sk.SecretKey, crp, c.ckgShare)
}

func (c *CellCNNProtocol) CKGGetShare() (*drlwe.CKGShare){
	return c.ckgShare
}

func (c *CellCNNProtocol) CKGAggregateShares(share *drlwe.CKGShare){
	c.ckgProtocol.AggregateShares(c.ckgShare, share, c.ckgShare)
}

func (c *CellCNNProtocol) CKGGenPublicKey(crp *ring.Poly){
	c.pk = ckks.NewPublicKey(c.params)
	c.ckgProtocol.GenCKKSPublicKey(c.ckgShare, crp, c.pk)
}

func (c *CellCNNProtocol) CKGWipe(){
	c.ckgProtocol = nil
	c.ckgShare = nil
}

// ==============================
// Relinearization Key Generation
// ==============================

func (c *CellCNNProtocol) NewRKGProtocol(){
	c.rkgProtocol = dckks.NewRKGProtocol(c.params)
	c.rkgEphemSk, c.rkgShareOne, c.rkgShareTwo = c.rkgProtocol.AllocateShares()
}

func (c *CellCNNProtocol) RKGRoundOne(crp []*ring.Poly){
	c.rkgProtocol.GenShareRoundOne(&c.sk.SecretKey, crp, c.rkgEphemSk, c.rkgShareOne)
}

func (c *CellCNNProtocol) RKGRoundTwo(crp []*ring.Poly){
	c.rkgProtocol.GenShareRoundTwo(c.rkgEphemSk, &c.sk.SecretKey, c.rkgShareOne, crp, c.rkgShareTwo)
}

func (c *CellCNNProtocol) CKGAggregateRoundOne(share *drlwe.RKGShare){
	c.rkgProtocol.AggregateShares(c.rkgShareOne, share, c.rkgShareOne)
}

func (c *CellCNNProtocol) CKGAggregateRoundTwo(share *drlwe.RKGShare){
	c.rkgProtocol.AggregateShares(c.rkgShareTwo, share, c.rkgShareTwo)
}

func (c *CellCNNProtocol) CKGGetShareOne() (*drlwe.RKGShare){
	return c.rkgShareOne
}

func (c *CellCNNProtocol) CKGGetShareTwo() (*drlwe.RKGShare){
	return c.rkgShareTwo
}

func (c *CellCNNProtocol) RKGGenRelinearizationKey(){
	c.rlk = ckks.NewRelinearizationKey(c.params)
	c.rkgProtocol.GenCKKSRelinearizationKey(c.rkgShareOne, c.rkgShareTwo, c.rlk)
}

func (c *CellCNNProtocol) RKGWipe(){
	c.rkgProtocol = nil
	c.rkgEphemSk = nil
	c.rkgShareOne = nil
	c.rkgShareTwo = nil
}

// ========================
// Rotation Keys Generation
// ========================

func (c *CellCNNProtocol) NewRTGProtocol(){

	if c.rotKey == nil{
		c.rotKey = new(ckks.RotationKeySet)
		c.rotKey.Keys = make(map[uint64]*rlwe.SwitchingKey)
	}

	c.rtgProtocol = dckks.NewRotKGProtocol(c.params)
	c.rtgShare = c.rtgProtocol.AllocateShares()
}

func (c *CellCNNProtocol) RTGGenShare(galEl uint64, crp []*ring.Poly){
	c.rtgProtocol.GenShare(&c.sk.SecretKey, galEl, crp, c.rtgShare)
}

func (c *CellCNNProtocol) RTGGetShare()(*drlwe.RTGShare){
	return c.rtgShare
}

func (c *CellCNNProtocol) RTGAggregate(share *drlwe.RTGShare){
	c.rtgProtocol.Aggregate(c.rtgShare, share, c.rtgShare)
}

func (c *CellCNNProtocol) RTGGenRotationKey(galEl uint64, crp []*ring.Poly){
	
	if c.rotKey.Keys[galEl] == nil{
		c.rotKey.Keys[galEl] = rlwe.NewSwitchingKey(c.params.N(), c.params.QPiCount(), c.params.Beta())
	}

	c.rtgProtocol.GenRotationKey(c.rtgShare, crp, c.rotKey.Keys[galEl])
}

func (c *CellCNNProtocol) RTGWipe(){
	c.rtgProtocol = nil
	c.rtgShare = nil
}

// ======================
// Key Switching Protocol
// ======================


func (c *CellCNNProtocol) NewCKSProtocol(){
	c.cksProtocol = dckks.NewCKSProtocol(c.params, 3.19)
	c.cksShare = c.cksProtocol.AllocateShare()
}

func (c *CellCNNProtocol) CKSGenShare(ciphertext *ckks.Ciphertext, targetKey *ring.Poly){
	c.cksProtocol.GenShare(c.sk.SecretKey.Value, targetKey, ciphertext, c.cksShare)
}

func (c *CellCNNProtocol) CKSGetShare() (dckks.CKSShare){
	return c.cksShare
}

func (c *CellCNNProtocol) CKSAggregate(share dckks.CKSShare){
	c.cksProtocol.AggregateShares(c.cksShare, share, c.cksShare)
}

func (c *CellCNNProtocol) CKSKeySwitchToPlaintext(ciphertext *ckks.Ciphertext) (*ckks.Plaintext){
	pt := ckks.NewCiphertext(c.params, 0, ciphertext.Level(), ciphertext.Scale())
	c.cksProtocol.KeySwitch(c.cksShare, ciphertext, pt)
	return pt.Plaintext()
}
