package cellCNN

import(
	"github.com/ldsec/lattigo/v2/ckks"
	"fmt"
	"math"
)


func RepackBeforeBootstrapping(ctU, ctPpool, ctW *ckks.Ciphertext, cells, filters, classes int, eval ckks.Evaluator, params *ckks.Parameters, sk *ckks.SecretKey){

	// ctU
	//
	// [[      classes      ] [ available ] [ garbage ]]
	//  | classes * filters | |           | | filters |
	//

	// ctPpool
	//
	// [[    available    ] [      garbage      ] [ #filters  ...  #filters ] [     garbage       ] [ available ]]
	//  | classes*filters | | (cells-1)*filters | |   classes * filters     | | (cells-1)*filters | |           | 
	//

	eval.Rotate(ctPpool, -((cells-1)*filters+classes*filters), ctPpool)

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

	//decryptPrint(2*(cells-1)*filters + 2*classes*filters, 2*(cells-1)*filters + 2*classes*filters+classes*filters, ctWRotate, params, sk)

	// Returns
	//  |========CTU========| |==============================CTPpool================================| |===========CTW===========|    
	// [[      classes      ] [      garbage      ] [ #filters  ...  #filters ] [     garbage       ] [ W transpose row encoded ] [ available ] [ garbage ]] 
	//  | classes * filters | | (cells-1)*filters | |   classes * filters     | | (cells-1)*filters | |    classes * filters    | |           | | filters |
	//
	
	eval.Add(ctU, ctPpool, ctU)
	eval.Add(ctU, ctWRotate, ctU)
}

func DummyBoot(ciphertext *ckks.Ciphertext, cells, features, filters, classes int, learningRate float64, params *ckks.Parameters, sk *ckks.SecretKey) (*ckks.Ciphertext){

	print := false

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

	idx := classes *filters

	//[[        U        ][             U            ] [ available ]]
	// | classes*filters || classes*filters*features | |           | 
	for i := 0; i < classes; i++ {
		c := complex(real(v[i*filters]), 0)
		for j := 0; j < filters*features; j++ {
			newv[idx + i*filters*features + j] = c
		}
	}

	idx += classes * filters*features
	
	//[[        U        ][            U             ] [       Ppool        ] [ available ]]
	// | classes*filters || classes*filters*features | | classes * filters  | |           |

	for i := 0; i < classes; i++ {
		for j := 0; j < filters; j++{
			newv[idx + i*filters + j] = complex(real(v[classes*filters + (cells-1)*filters + i * filters + j]) * learningRate, 0)
		}
	}

	idx += classes * filters

	//[[        U        ][            U             ] [       Ppool        ] [   W transpose row encoded    ] [ available ]]
	// | classes*filters || classes*filters*features | | classes * filters  | | classes * filters * features |
	
	for i := 0; i < classes; i++ {
		for j := 0; j < filters; j++ {
			c := complex(real(v[(2*(cells-1)*filters + 2*classes*filters) + i * filters + j]), 0) * complex(math.Pow(0.5 * learningRate / float64(cells), 0.5), 0)
			for k := 0 ; k < features; k++ {
				newv[idx + i*filters*features + k*filters + j] = c
			}
		}
	}

	idx += classes * filters * features

	if print {
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