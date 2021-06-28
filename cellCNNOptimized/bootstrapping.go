package cellCNN

import (
	"fmt"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"math"
)

func RepackBeforeBootstrapping(ctU, ctPpool, ctW, ctDWprev, ctDCprev *ckks.Ciphertext, cells, Filters, Classes int, eval ckks.Evaluator, params *ckks.Parameters, sk *rlwe.SecretKey) {

	// ctU
	//
	// [[      Classes      ] [ available ] [   garbage  ]]
	//  | Classes * Filters | |           | | Filters -1 |
	//

	// ctPpool
	//
	// [[    available    ] [      garbage      ] [ #Filters  ...  #Filters ] [     garbage       ] [ available ]]
	//  | Classes*Filters | | (cells-1)*Filters | |   Classes * Filters     | | (cells-1)*Filters | |           |
	//

	eval.Rotate(ctPpool, -((cells-1)*Filters + Classes*Filters), ctPpool)
	eval.Add(ctU, ctPpool, ctU)

	// ctW
	//
	// [[                 available               ] [ W transpose row encoded ] [ available ]]
	//  | 2*(Classes*Filters + (cells-1)*Filters) | |    Classes * Filters    | |           |
	//

	ctWRotate := eval.RotateNew(ctW, -(2*(cells-1)*Filters + 2*Classes*Filters))

	eval.MultByConst(ctWRotate, ctU.Scale()/ctWRotate.Scale(), ctWRotate)

	if err := eval.Rescale(ctWRotate, params.Scale(), ctWRotate); err != nil {
		panic(err)
	}

	ctWRotate.SetScale(ctU.Scale())
	eval.Add(ctU, ctWRotate, ctU)

	if ctDWprev != nil && ctDWprev != nil {

		eval.Rotate(ctDWprev, -(2*(cells-1)*Filters + 3*Classes*Filters), ctDWprev)
		eval.Rotate(ctDCprev, -(2*(cells-1)*Filters + 4*Classes*Filters), ctDCprev)

		eval.Add(ctDWprev, ctDCprev, ctDWprev)

		eval.MultByConst(ctDWprev, ctU.Scale()/ctDWprev.Scale(), ctDWprev)

		if err := eval.Rescale(ctDWprev, params.Scale(), ctDWprev); err != nil {
			panic(err)
		}

		ctDWprev.SetScale(ctU.Scale())
		eval.Add(ctU, ctDWprev, ctU)
	}

	//decryptPrint(2*(cells-1)*Filters + 2*Classes*Filters, 2*(cells-1)*Filters + 2*Classes*Filters+Classes*Filters, ctWRotate, params, sk)

	// Returns
	//  [========CTU========] [==============================CTPpool================================] [===========CTW===========] [     prevCTDW      ] [                prevCTDC                  ] [available] [ garbage ]
	//  | Classes * Filters | | (cells-1)*Filters | |   Classes * Filters     | | (cells-1)*Filters | |    Classes * Filters    | | Classes * Filters | [Classes * Filters * (cells + Features + 1)] |         | | Filters |
	//
}

func DummyBoot(ciphertext *ckks.Ciphertext, cells, Features, Filters, Classes int, LearningRate, Momentum float64, params ckks.Parameters, sk *rlwe.SecretKey) *ckks.Ciphertext {

	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptorFromSk(params, sk)

	v := encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())

	newv := make([]complex128, params.Slots())

	//[[        U        ][ available ]]
	// | Classes*Filters ||           |

	for i := 0; i < Classes; i++ {
		c := complex(real(v[i*Filters]), 0)
		for j := 0; j < Filters; j++ {
			newv[i*Filters+j] = c
		}
	}

	idx := Classes * Filters

	//[[        U        ][                      U                     ] [ available ]]
	// | Classes*Filters || Classes * Filters * (cells + Features + 1) | |           |
	for i := 0; i < Classes; i++ {
		c := complex(real(v[i*Filters]), 0)
		for j := 0; j < Filters*(cells+Features+1); j++ {
			newv[idx+i*Filters*(cells+Features+1)+j] = c
		}
	}

	idx += Classes * Filters * (cells + Features + 1)

	//[[        U        ][                      U                     ] [       Ppool        ] [ available ]]
	// | Classes*Filters || Classes * Filters * (cells + Features + 1) | | Classes * Filters  | |           |

	for i := 0; i < Classes; i++ {
		for j := 0; j < Filters; j++ {
			newv[idx+i*Filters+j] = complex(real(v[Classes*Filters+(cells-1)*Filters+i*Filters+j])*LearningRate, 0)
		}
	}

	idx += Classes * Filters

	//[[        U        ][                                            ] [       Ppool        ] [          W transpose row encoded           ] [ available ]]
	// | Classes*Filters || Classes * Filters * (cells + Features + 1) | | Classes * Filters  | | Classes * Filters * (cells + Features + 1) |

	for i := 0; i < Classes; i++ {

		for j := 0; j < Filters*(cells+Features+1); j++ {

			c := real(v[(2*(cells-1)*Filters+2*Classes*Filters)+i*Filters+(j%Filters)])
			c *= math.Pow(LearningRate/float64(cells), 0.5)

			newv[idx+i*Filters*(cells+Features+1)+j] = complex(c, 0)
		}
	}

	idx += Classes * Filters * (cells + Features + 1)

	//[[        U        ][                      U                     ] [       Ppool        ] [           W transpose row encoded         ] [  Previous DeltaW  ] [           Previous DeltaC                ] [ available ]]
	// | Classes*Filters || Classes * Filters * (cells + Features + 1) | | Classes * Filters  | | Classes * Filters * (cells + Features + 1)| [ Classes * Filters ] [Classes * Filters * (cells + Features + 1)]

	for i := 0; i < Classes*Filters; i++ {
		newv[idx+i] = complex(real(v[(2*(cells-1)*Filters+3*Classes*Filters)+i])*Momentum, 0)
	}

	idx += Classes * Filters

	for i := 0; i < Filters*(cells+Features+1); i++ {
		newv[idx+i] = complex(real(v[(2*(cells-1)*Filters+4*Classes*Filters)+i])*Momentum, 0)
	}

	idx += Filters * (cells + Features + 1)

	if false {
		fmt.Println("Repacked Plaintext")
		for i := 0; i < idx; i++ {
			fmt.Println(i, newv[i])
		}
	}

	pt := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	encoder.EncodeNTT(pt, newv, params.LogSlots())
	newCt := encryptor.EncryptNew(pt)

	return newCt

}

func RepackBeforeBootstrappingWithPrepooling(ctU, ctPpool, ctW, ctDCprev, ctDWprev *ckks.Ciphertext, BatchSize, Filters, Classes int, eval ckks.Evaluator) {

	denseMatrixSize := DenseMatrixSize(Filters, Classes)

	// ctU
	//
	// [[  S0U0   ] ... [  SiU0   ] [  S0U1   ] ... [  SiU1   ] [ available ] [   garbage  ]]
	//  | Filters | ... | Filters | | Filters | ... | Filters | |           | | Filters -1 |
	//  |    Filters * batches     | |    Filters * batches     |
	//  |				batches * DenseMatrixSize       	      |

	// ctPpool
	//
	// [[         available         ] [   S0P   ] ... [   S1P   ] [ available ]]
	//  | batches * DenseMatrixSize | | Filters | ... | Filters | |           |
	//	                              |    batches * Filters     |
	//

	eval.Add(ctU, eval.RotateNew(ctPpool, -BatchSize*denseMatrixSize), ctU)

	// ctW
	//
	// [[               available               ] [     W transpose row encoded       ] [ available ]]
	//  | batches * (DensematrixSize + Filters) | |    batches *  DenseMatrixSize     | |           |
	//

	ctWRotate := eval.RotateNew(ctW, -2*BatchSize*denseMatrixSize)
	eval.Add(ctU, ctWRotate, ctU)

	// ctDWprev
	//
	// [[          available          ] [ W transpose row encoded ] [ available ]]
	//  | 3*batches * DenseMatrixSize | | batches*DenseMatrixSize | |           |
	//

	// ctDCprev
	//
	// [[          available          ] [                          C                               ] [ available ]]
	//  | 4*batches * DenseMatrixSize | | batches * Filters + (Features/2 -1)*2*Filters + Filters  | |           |
	//

	if ctDCprev != nil && ctDCprev != nil {
		eval.Rotate(ctDWprev, -3*BatchSize*denseMatrixSize, ctDWprev)
		eval.Rotate(ctDCprev, -4*BatchSize*denseMatrixSize, ctDCprev)
		eval.Add(ctDWprev, ctDCprev, ctDWprev)
		eval.Add(ctU, ctDWprev, ctU)
	}

	// CTU : lvl2
	// CTPool : lvl3
	// CTW : lvl3
	// CTWPrev : lvl5
	// CTCPrev : lvl4

	// Returns
	//  [            CTU            ] [     CTPpool       ] [             CTW            ] [         prevCTDW           ] [                         prevCTDC                        ] [available] [   garbage  ]
	//  | batches * DenseMatrixSize | | batches * Filters | | batches *  DenseMatrixSize | | batches *  DenseMatrixSize | | batches * Filters + (Features/2 -1)*2*Filters + Filters | |         | | Filters -1 |
	//
}
