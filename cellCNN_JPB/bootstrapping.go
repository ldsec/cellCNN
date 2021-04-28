package cellCNN

import(
	"github.com/ldsec/lattigo/v2/ckks"
	"math"
	"fmt"
)


func RepackBeforeBootstrapping(ctU, ctPpool, ctW, ctDWprev, ctDCprev *ckks.Ciphertext, cells, filters, classes int, eval ckks.Evaluator, params *ckks.Parameters, sk *ckks.SecretKey){

	// ctU
	//
	// [[      classes      ] [ available ] [   garbage  ]]
	//  | classes * filters | |           | | filters -1 |
	//

	// ctPpool
	//
	// [[    available    ] [      garbage      ] [ #filters  ...  #filters ] [     garbage       ] [ available ]]
	//  | classes*filters | | (cells-1)*filters | |   classes * filters     | | (cells-1)*filters | |           | 
	//

	eval.Rotate(ctPpool, -((cells-1)*filters+classes*filters), ctPpool)
	eval.Add(ctU, ctPpool, ctU)

	// ctW
	//
	// [[                 available               ] [ W transpose row encoded ] [ available ]]
	//  | 2*(classes*filters + (cells-1)*filters) | |    classes * filters    | |           |
 	//

	ctWRotate := eval.RotateNew(ctW, -(2*(cells-1)*filters + 2*classes*filters))

	eval.MultByConst(ctWRotate, ctU.Scale()/ctWRotate.Scale(), ctWRotate)

	if err := eval.Rescale(ctWRotate, params.Scale(), ctWRotate); err != nil {
		panic(err)
	}

	ctWRotate.SetScale(ctU.Scale())
	eval.Add(ctU, ctWRotate, ctU)

	if ctDWprev != nil && ctDWprev != nil{

		eval.Rotate(ctDWprev, -(2*(cells-1)*filters + 3*classes*filters), ctDWprev)
		eval.Rotate(ctDCprev, -(2*(cells-1)*filters + 4*classes*filters), ctDCprev)

		eval.Add(ctDWprev, ctDCprev, ctDWprev)

		eval.MultByConst(ctDWprev, ctU.Scale()/ctDWprev.Scale(), ctDWprev)

		if err := eval.Rescale(ctDWprev, params.Scale(), ctDWprev); err != nil {
			panic(err)
		}

		ctDWprev.SetScale(ctU.Scale())
		eval.Add(ctU, ctDWprev, ctU)
	}

	//decryptPrint(2*(cells-1)*filters + 2*classes*filters, 2*(cells-1)*filters + 2*classes*filters+classes*filters, ctWRotate, params, sk)

	// Returns
	//  [========CTU========] [==============================CTPpool================================] [===========CTW===========] [     prevCTDW      ] [                prevCTDC                  ] [available] [ garbage ]
	//  | classes * filters | | (cells-1)*filters | |   classes * filters     | | (cells-1)*filters | |    classes * filters    | | classes * filters | [classes * filters * (cells + features + 1)] |         | | filters |
	//
}

func DummyBoot(ciphertext *ckks.Ciphertext, cells, features, filters, classes int, learningRate, momentum float64, params *ckks.Parameters, sk *ckks.SecretKey) (*ckks.Ciphertext){

	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptorFromSk(params, sk)

	v := encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())

	newv := make([]complex128, params.Slots())

	//[[        U        ][ available ]]
	// | classes*filters ||           | 

	for i := 0; i < classes; i++ {
		c := complex(real(v[i*filters]), 0)
		for j := 0; j < filters; j++ {
			newv[i*filters + j] = c
		}
	}

	idx := classes*filters

	//[[        U        ][                      U                     ] [ available ]]
	// | classes*filters || classes * filters * (cells + features + 1) | |           | 
	for i := 0; i < classes; i++ {
		c := complex(real(v[i*filters]), 0)
		for j := 0; j < filters * (cells + features + 1); j++ {
			newv[idx + i*filters*(cells + features + 1) + j] = c
		}
	}

	idx += classes * filters * (cells + features + 1)
	
	//[[        U        ][                      U                     ] [       Ppool        ] [ available ]]
	// | classes*filters || classes * filters * (cells + features + 1) | | classes * filters  | |           |

	for i := 0; i < classes; i++ {
		for j := 0; j < filters; j++{
			newv[idx + i*filters + j] = complex(real(v[classes*filters + (cells-1)*filters + i * filters + j]) * learningRate, 0)
		}
	}

	idx += classes * filters

	//[[        U        ][                                            ] [       Ppool        ] [          W transpose row encoded           ] [ available ]]
	// | classes*filters || classes * filters * (cells + features + 1) | | classes * filters  | | classes * filters * (cells + features + 1) |
	

	for i := 0; i < classes; i++ {

		for j := 0; j < filters * (cells + features + 1); j++ {

			c := real(v[(2*(cells-1)*filters + 2*classes*filters) + i * filters + (j%filters)])
			c *= math.Pow(learningRate / float64(cells), 0.5)

			newv[idx + i*filters*(cells + features + 1) + j] = complex(c, 0)
		}
	}

	idx += classes * filters * (cells + features + 1)


	//[[        U        ][                      U                     ] [       Ppool        ] [           W transpose row encoded         ] [  Previous DeltaW  ] [           Previous DeltaC                ] [ available ]]
	// | classes*filters || classes * filters * (cells + features + 1) | | classes * filters  | | classes * filters * (cells + features + 1)| [ classes * filters ] [classes * filters * (cells + features + 1)]
	

	for i := 0; i < classes * filters; i++ {
		newv[idx + i] = complex(real(v[(2*(cells-1)*filters + 3*classes*filters)+i])*momentum, 0)
	}

	idx += classes*filters

	for i := 0; i < filters * (cells + features + 1); i++ {
		newv[idx + i] = complex(real(v[(2*(cells-1)*filters + 4*classes*filters)+i])*momentum, 0)
	}

	idx += filters * (cells + features + 1)

	if false {
		fmt.Println("Repacked Plaintext")
		for i := 0; i < idx; i++{
			fmt.Println(i, newv[i])
		}
	}
	

	pt := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	encoder.EncodeNTT(pt, newv, params.LogSlots())
	newCt := encryptor.EncryptNew(pt)

	return newCt

}

func RepackBeforeBootstrappingWithPrepooling(ctU, ctPpool, ctW, ctDCprev, ctDWprev *ckks.Ciphertext, batchSize, filters, classes int, eval ckks.Evaluator, params *ckks.Parameters, sk *ckks.SecretKey){

	denseMatrixSize := DenseMatrixSize(filters, classes)

	// ctU
	//
	// [[   S0U0   ] ... [   SiU0   ] [   S0U1   ] ... [   SiU1   ] [ available ] [   garbage  ]]
	//  | features | ... | features | | features | ... | features | |           | | filters -1 |
	//  |    features * batches     | |    features * batches     |
	//  |				batches * DenseMatrixSize       	      |

	// ctPpool
	//
	// [[         available         ] [    S0P   ] ... [    S1P   ] [ available ]]
	//  | batches * DenseMatrixSize | | features | ... | features | |           | 
	//	                              |    batches * features     |
	//

	eval.Add(ctU, eval.RotateNew(ctPpool, -batchSize*denseMatrixSize), ctU)

	// ctW
	//
	// [[                available               ] [     W transpose row encoded       ] [ available ]]
	//  | batches * (DensematrixSize + features) | |    batches *  DenseMatrixSize     | |           |
 	//

	ctWRotate := eval.RotateNew(ctW, -2*batchSize*denseMatrixSize)
	eval.Add(ctU, ctWRotate, ctU)

	// ctDWprev
	//
	// [[          available          ] [ W transpose row encoded ] [ available ]]
	//  | 3*batches * DenseMatrixSize | | batches*DenseMatrixSize | |           |
 	//

 	// ctDCprev
	//
	// [[          available          ] [                          C                               ] [ available ]]
	//  | 4*batches * DenseMatrixSize | | batches * filters + (features/2 -1)*2*filters + filters  | |           |
 	//

	eval.Rotate(ctDWprev, -3*batchSize*denseMatrixSize, ctDWprev)
	eval.Rotate(ctDCprev, -4*batchSize*denseMatrixSize, ctDCprev)

	eval.Add(ctDWprev, ctDCprev, ctDWprev)
	eval.Add(ctU, ctDWprev, ctU)

	// CTU : lvl2
	// CTPool : lvl3
	// CTW : lvl3
	// CTWPrev : lvl4
	// CTCPrev : lvl5

	// Returns
	//  [            CTU            ] [      CTPpool       ] [             CTW            ] [         prevCTDW           ] [                         prevCTDC                        ] [available] [   garbage  ]
	//  | batches * DenseMatrixSize | | batches * features | | batches *  DenseMatrixSize | | batches *  DenseMatrixSize | | batches * filters + (features/2 -1)*2*filters + filters | |         | | filters -1 |
	//
}

func DummyBootWithPrepooling(ciphertext *ckks.Ciphertext, batchSize, features, filters, classes int, learningRate, momentum float64, params *ckks.Parameters, sk *ckks.SecretKey) (*ckks.Ciphertext){

	//  [            CTU            ] [      CTPpool       ] [             CTW            ] [         prevCTDW           ] [                         prevCTDC                        ] [available] [   garbage  ]
	//  | batches * DenseMatrixSize | | batches * features | | batches *  DenseMatrixSize | | batches *  DenseMatrixSize | | batches * filters + (features/2 -1)*2*filters + filters | |         | | filters -1 |
	//

	//DecryptPrint(2*batchSize, filters, true, ciphertext, params, sk)

	convolutionMatrixSize := ConvolutionMatrixSize(batchSize, features, filters)
	denseMatrixSize := DenseMatrixSize(filters, classes)

	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptorFromSk(params, sk)

	v := encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())

	newv := make([]complex128, params.Slots())

	//[[            U            ] [ available ]]
	// | batches*classes*filters | |           | 

	// Reorder the slots (groupes samples together) from
	//
	// [ S0U0 ] ... [ SiU0 ] [ S0U1 ] ... [ SiU1 ] ]
	//
	// to
	//
	// [ S0U0 ] [ S0U1 ] ... [ SiU0 ] ... [ SiU1 ] ]
	// 
	for i := 0; i < batchSize; i++ {
		for j := 0; j < classes; j++{
			c := complex(real(v[i*filters+j*filters*batchSize]), 0)
			for k := 0; k < filters; k++{
				newv[i*filters*classes + j*filters + k] = c
			}
		}	
	}

	idx := batchSize*denseMatrixSize

	

	//[[            U            ] [                U                ] [ available ]]
	// | batches*classes*filters | | batches * convolutionMatrixSize | |           | 
	for i := 0; i < classes; i++ {
		c := complex(real(v[i*filters]), 0)
		for j := 0; j < convolutionMatrixSize; j++ {
			newv[idx + i*convolutionMatrixSize + j] = c
		}
	}

	idx += classes * convolutionMatrixSize

	
	
	//[[            U            ] [                U                ] [         Ppool           ] [ available ]]
	// | batches*classes*filters | | batches * convolutionMatrixSize | | batches*classes*filters | |           |

	// Reorder the slots (groupes samples together) from
	//
	// [ S0P0 ] ... [ SiP0 ] ... [ S0P0 ] ... [ SiP0 ]
	// to
	// [ S0P0 ] ... [ S0P0 ] ... [ SiP0 ] ... [ SiP0 ]
	// | classes * filters | ... | classes * filters |
	for i := 0; i < batchSize; i++{
		for j := 0; j < filters; j++ {
			c := complex(real(v[batchSize*denseMatrixSize + i*filters + j]) * learningRate, 0)
			for k := 0; k < classes; k++ {
				newv[idx + i*filters*classes + k*filters + j] = c
			}
		}
	}
	

	idx += batchSize * denseMatrixSize

	if false {
		fmt.Println("Repacked Plaintext")
		for i := 0; i < idx; i++{
			fmt.Println(i, newv[i])
		}
	}



	//[[            U            ] [                U                ] [         Ppool           ] [    W transpose row encoded      ] [ available ]]
	// | batches*classes*filters | | batches * convolutionMatrixSize | | batches*classes*filters | | classes * convolutionMatrixSize | |           |
	for i := 0; i < classes; i++ {

		for j := 0; j < convolutionMatrixSize; j++ {

			c := real(v[2*denseMatrixSize + i * filters + (j%filters)])

			newv[idx + convolutionMatrixSize*i + j] = complex(c, 0)
		}
	}

	idx += classes * convolutionMatrixSize




	//[[        U        ][                U                ] [       Ppool        ] [      W transpose row encoded    ] [  Previous DeltaW  ] [     Previous DeltaC           ] [ available ]]
	// | classes*filters || classes * ConvolutionMatrixSize | | classes * filters  | | classes * convolutionMatrixSize | [ classes * filters ] [classes * convolutionMatrixSize]
	

	for i := 0; i < classes * filters; i++ {
		newv[idx + i] = complex(real(v[3*denseMatrixSize+i])*momentum, 0)
	}

	idx += denseMatrixSize

	for i := 0; i < filters * (features + 2); i++ {
		newv[idx + i] = complex(real(v[4*denseMatrixSize+i])*momentum, 0)
	}

	idx += convolutionMatrixSize

	
	
	
	pt := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	encoder.EncodeNTT(pt, newv, params.LogSlots())
	newCt := encryptor.EncryptNew(pt)

	//DecryptPrint(2*batchSize, filters, true, newCt, params, sk)

	return newCt

}