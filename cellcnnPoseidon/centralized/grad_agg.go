package centralized

import (
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
)

// Gradients an object to aggregate the gradients of conv and dense
type Gradients struct {
	filters []*ckks.Ciphertext
	dense   *ckks.Ciphertext
}

// NewGradient initialize a new gradient,
// when unmarshall first n-1 for filters and last one for dense
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

// Aggregate conduct: self = self + data
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

// Bootstrapping use sk to re-encrypt the ciphertext for dummy bootstrapping
func (g *Gradients) DummyBootstrapping(encoder ckks.Encoder, params ckks.Parameters, sk *rlwe.SecretKey) {
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

// GetGradientBinary return the byte representation, first n-1 for filters, last one for dense
func (g *Gradients) GetGradientBinary() [][]byte {
	res := make([][]byte, len(g.filters)+1)
	var err error = nil
	for i, each := range g.filters {
		res[i], err = each.MarshalBinary()
		if err != nil {
			panic("err in GetGradientBinary filters")
		}
	}
	res[len(res)-1], err = g.dense.MarshalBinary()
	if err != nil {
		panic("err in GetGradientBinary dense")
	}
	return res
}

// LoadGradientBinary first n-1 for filters, last one for dense
func (g *Gradients) LoadGradientBinary(data [][]byte) {
	for i := 0; i < len(data)-1; i++ {
		g.filters[i] = new(ckks.Ciphertext)
		if err := g.filters[i].UnmarshalBinary(data[i]); err != nil {
			panic("fail to LoadGradientBinary Gradients filters")
		}
	}
	g.dense = new(ckks.Ciphertext)
	if err := g.dense.UnmarshalBinary(data[len(data)-1]); err != nil {
		panic("fail to LoadGradientBinary Gradients filters")
	}
}

// GetPlaintext for debug only, decrypt a ciphertext according to idx.
// return return the slots at certain indices according to inds.
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

// GetFilters return the filters of the gradient
func (g *Gradients) GetFilters() []*ckks.Ciphertext {
	return g.filters
}
