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
	c.CkgProtocol = dckks.NewCKGProtocol(c.params)
	c.CkgShare = c.CkgProtocol.AllocateShares()
}

func (c *CellCNNProtocol) CKGGenShare(crp *ring.Poly){
	c.CkgProtocol.GenShare(c.sk, crp, c.CkgShare)
}

func (c *CellCNNProtocol) CKGGetShare() (*drlwe.CKGShare){
	return c.CkgShare
}

func (c *CellCNNProtocol) CKGAggregateShares(A, B, C *drlwe.CKGShare){
	c.CkgProtocol.AggregateShares(A, B, C)
}

func (c *CellCNNProtocol) CKGGenPublicKey(share *drlwe.CKGShare, crp *ring.Poly){
	c.pk = ckks.NewPublicKey(c.params)
	c.CkgProtocol.GenPublicKey(share, crp, c.pk)
}

func (c *CellCNNProtocol) CKGWipe(){
	c.CkgProtocol = nil
	c.CkgShare = nil
}

// ==============================
// Relinearization Key Generation
// ==============================

func (c *CellCNNProtocol) NewRKGProtocol(){
	c.RkgProtocol = dckks.NewRKGProtocol(c.params)
	c.RkgEphemSk, c.RkgShareOne, c.RkgShareTwo = c.RkgProtocol.AllocateShares()
}

func (c *CellCNNProtocol) RKGRoundOne(crp []*ring.Poly){
	c.RkgProtocol.GenShareRoundOne(c.sk, crp, c.RkgEphemSk, c.RkgShareOne)
}

func (c *CellCNNProtocol) RKGRoundTwo(shareOne *drlwe.RKGShare, crp []*ring.Poly){
	c.RkgProtocol.GenShareRoundTwo(c.RkgEphemSk, c.sk, shareOne, crp, c.RkgShareTwo)
}

func (c *CellCNNProtocol) CKGAggregate(A, B, C *drlwe.RKGShare){
	c.RkgProtocol.AggregateShares(A, B, C)
}

func (c *CellCNNProtocol) CKGGetShareOne() (*drlwe.RKGShare){
	return c.RkgShareOne
}

func (c *CellCNNProtocol) CKGGetShareTwo() (*drlwe.RKGShare){
	return c.RkgShareTwo
}

func (c *CellCNNProtocol) RKGGenRelinearizationKey(shareOne, shareTwo *drlwe.RKGShare){
	c.rlk = ckks.NewRelinearizationKey(c.params)
	c.RkgProtocol.GenRelinearizationKey(shareOne, shareTwo, c.rlk)
}

func (c *CellCNNProtocol) RKGWipe(){
	c.RkgProtocol = nil
	c.RkgEphemSk = nil
	c.RkgShareOne = nil
	c.RkgShareTwo = nil
}

// ========================
// Rotation Keys Generation
// ========================

func (c *CellCNNProtocol) NewRTGProtocol(){

	if c.rotKey == nil{
		c.rotKey = new(rlwe.RotationKeySet)
		c.rotKey.Keys = make(map[uint64]*rlwe.SwitchingKey)
	}

	c.RtgProtocol = dckks.NewRotKGProtocol(c.params)
	c.RtgShare = c.RtgProtocol.AllocateShares()
}

func (c *CellCNNProtocol) RTGGenShare(galEl uint64, crp []*ring.Poly){
	c.RtgProtocol.GenShare(c.sk, galEl, crp, c.RtgShare)
}

func (c *CellCNNProtocol) RTGGetShare()(*drlwe.RTGShare){
	return c.RtgShare
}

func (c *CellCNNProtocol) RTGAggregate(A, B, C *drlwe.RTGShare){
	c.RtgProtocol.Aggregate(A, B, C)
}

func (c *CellCNNProtocol) RTGGenRotationKey(galEl uint64, crp []*ring.Poly, share *drlwe.RTGShare){
	
	if c.rotKey.Keys[galEl] == nil{
		c.rotKey.Keys[galEl] = rlwe.NewSwitchingKey(c.params.Parameters)
	}

	c.RtgProtocol.GenRotationKey(share, crp, c.rotKey.Keys[galEl])
}

func (c *CellCNNProtocol) RTGWipe(){
	c.RtgProtocol = nil
	c.RtgShare = nil
}

// ======================
// Key Switching Protocol
// ======================


func (c *CellCNNProtocol) NewCKSProtocol(){
	c.CksProtocol = dckks.NewCKSProtocol(c.params, 3.19)
	c.CksShare = c.CksProtocol.AllocateShare()
}

func (c *CellCNNProtocol) CKSGenShare(ciphertext *ckks.Ciphertext, targetKey *ring.Poly){
	c.CksProtocol.GenShare(c.sk.Value, targetKey, ciphertext, c.CksShare)
}

func (c *CellCNNProtocol) CKSGetShare() (dckks.CKSShare){
	return c.CksShare
}

func (c *CellCNNProtocol) CKSAggregate(A, B, C dckks.CKSShare){
	c.CksProtocol.AggregateShares(A, B, C)
}
