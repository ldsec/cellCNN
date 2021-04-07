package cellCNN

import(
	"github.com/ldsec/lattigo/v2/ckks"
)

func Forward(ptL []*ckks.Plaintext, ctC *ckks.Ciphertext, ctW *ckks.Ciphertext, cells, features, filters, classes int, eval ckks.Evaluator, params *ckks.Parameters, sk *ckks.SecretKey) (*ckks.Ciphertext) {
	ctP := Convolution(ptL, ctC, features, filters, eval, params, sk)


	ctPpool := Pooling(ctP, cells, filters, classes, eval)
	ctU := DenseLayer(ctPpool, ctW, filters, classes, eval)
	RepackBeforeBootstrapping(ctU, ctPpool, ctW, cells, filters, classes, eval, params, sk)
	return ctU
}

// =====================
// ==== Convolution ====
// =====================
//
// Returns
//
// [[ P = L X C row encoded ] [        available        ]]
//  |    cells * filters    | | Slots - cells * filters | 
//
func Convolution(L0 []*ckks.Plaintext, C *ckks.Ciphertext, features, filters int, eval ckks.Evaluator, params *ckks.Parameters, sk *ckks.SecretKey) (*ckks.Ciphertext){
	return  MulMatrixLeftPtWithRightCt(L0, C, features, filters, eval, params, sk)
}


// =====================
// ====== Pooling ======
// =====================
//
// Returns
//
// [[ #filters  ...  #filters ] [     garbage       ] [                available                  ] [      garbage      ]]
//  |   classes * filters     | | (cells-1)*filters | | Slots - filters * (classes + 2*cells - 2) | | (cells-1)*filters |
//
func Pooling(ct *ckks.Ciphertext, cells, filters, classes int, eval ckks.Evaluator) (*ckks.Ciphertext){

	rotHoisted := []int{}
	for i := 1; i < classes; i++ {
		rotHoisted = append(rotHoisted, -filters*i)
	}

	ctPpool := ct.CopyNew().Ciphertext()

	tmp := eval.RotateHoisted(ct, rotHoisted) 

	eval.InnerSum(ctPpool, filters, cells, ctPpool)

	for i := range tmp{
		eval.Add(ctPpool, tmp[i], ctPpool)
	}
	
	ctPpool.MulScale(float64(cells))

	return ctPpool
}


// =====================
// ======= Dense =======
// =====================
// Returns
//
// [[      classes      ] [        available          ] [ garbage ]]
//  | classes * filters | | Slots-(classes+1)*filters | | filters |
//
func DenseLayer(ctP, ctW *ckks.Ciphertext, filters, classes int, eval ckks.Evaluator) (*ckks.Ciphertext) {
	ctU := eval.MulRelinNew(ctP, ctW)
	eval.Rescale(ctU, ctW.Scale(), ctU)
	eval.InnerSum(ctU, 1, filters, ctU)
	return ctU
}