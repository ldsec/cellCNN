package centralized

import (
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
)

type Gradients struct {
	filters []*ckks.Ciphertext
	dense   *ckks.Ciphertext
}

func (g *Gradients) NewGradient(data [][]byte) {
	g.filters = make([]*ckks.Ciphertext, len(data)-1)
	for i, each := range data[:len(data)-1] {
		g.filters[i] = new(ckks.Ciphertext)
		if err := g.filters[i].UnmarshalBinary(each); err != nil {
			panic("fail to unmarshall Gradients")
		}
	}
	g.dense = new(ckks.Ciphertext)
	if err := g.dense.UnmarshalBinary(data[len(data)-1]); err != nil {
		panic("fail to unmarshall conv filter weights")
	}
}

// aggregate data to self
func (g *Gradients) Aggregate(data interface{}, eval ckks.Evaluator) {
	switch data := data.(type) {
	case [][]byte:
		// tmpFilters = make([]*ckks.Ciphertext, len(data)-1)
		for i, each := range data[:len(data)-1] {
			tmpFilter := new(ckks.Ciphertext)
			if err := tmpFilter.UnmarshalBinary(each); err != nil {
				panic("fail to unmarshall Gradients Aggregate")
			}
			eval.Add(g.filters[i], tmpFilter, g.filters[i])
		}
		tmpDense := new(ckks.Ciphertext)
		if err := tmpDense.UnmarshalBinary(data[len(data)-1]); err != nil {
			panic("fail to unmarshall conv filter weights")
		}
		eval.Add(g.dense, tmpDense, g.dense)
	case []*ckks.Ciphertext:
		// tmpFilters = make([]*ckks.Ciphertext, len(data)-1)
		for i, each := range data[:len(data)-1] {
			eval.Add(g.filters[i], each, g.filters[i])
		}
		eval.Add(g.dense, data[len(data)-1], g.dense)
	case *Gradients:
		for i, each := range data.filters {
			eval.Add(g.filters[i], each, g.filters[i])
		}
		eval.Add(g.dense, data.dense, g.dense)
	}
}

// bootstrap
func (g *Gradients) Bootstrapping(encoder ckks.Encoder, params ckks.Parameters, sk *rlwe.SecretKey) {
	ect := ckks.NewEncryptorFromSk(params, sk)
	dct := ckks.NewDecryptor(params, sk)

	// re-encrypt filters
	for i, each := range g.filters {
		plain := encoder.Decode(dct.DecryptNew(each), params.LogSlots())
		replain := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), plain, params.LogSlots())
		g.filters[i] = ect.EncryptNew(replain)
	}

	// re-encrypt dense
	plain := encoder.Decode(dct.DecryptNew(g.dense), params.LogSlots())
	replain := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), plain, params.LogSlots())
	g.dense = ect.EncryptNew(replain)
}

func (g *Gradients) Marshall() [][]byte {
	res := make([][]byte, len(g.filters)+1)
	var err error = nil
	for i, each := range g.filters {
		res[i], err = each.MarshalBinary()
		if err != nil {
			panic("err in marshall Gradients")
		}
	}
	res[len(res)-1], err = g.dense.MarshalBinary()
	if err != nil {
		panic("err in marshall Gradients")
	}
	return res
}

func (g *Gradients) Unmarshall(data [][]byte) []*ckks.Ciphertext {
	res := make([]*ckks.Ciphertext, len(data))
	for i, each := range data {
		res[i] = new(ckks.Ciphertext)
		if err := res[i].UnmarshalBinary(each); err != nil {
			panic("fail to unmarshall Gradients")
		}
	}
	return res
}

func (g *Gradients) GetPlaintext(idx int, inds []int, params ckks.Parameters, encoder ckks.Encoder, decryptor ckks.Decryptor) []complex128 {
	var ct *ckks.Ciphertext
	if idx < len(g.filters) {
		ct = g.filters[idx]
	} else {
		ct = g.dense
	}
	plaintext := encoder.Decode(decryptor.DecryptNew(ct), params.LogSlots())
	res := make([]complex128, len(inds))
	for i, each := range inds {
		res[i] = plaintext[each]
	}
	return res
}
