package decentralized

import "go.dedis.ch/kyber/v3/group/edwards25519"

// Suite is the type of keys used to secure communication in Onet
func GetSuit() *edwards25519.SuiteEd25519 {
	return edwards25519.NewBlakeSHA256Ed25519()
}

const approximationDegree uint = 3
const interval float64 = 3
const maxM1N2Ratio float64 = 8.0
const ncells int = 5
const nfilters int = 2
const nmakers int = 2
const nclasses int = 2
const nodeBatchSize int = 2
const learningRate float64 = 0.1
const momentum float64 = 1
const maxIterations int = 3

// const NCELLS = 300
// const NFEATURES = 16
// const NSAMPLES = 2250
// const NSAMPLES_DIST = 400
// const NCLASSES = 3
// const NFILTERS = 7

const HOSTS int = 3

// TODO:
// 1. set the correct lr sign
// 2. how to process batch backward with momentum:
//		batch forward
//		for conv1d: average pure gradients over the batch
// 		for dense: add pure gradients across the batch
