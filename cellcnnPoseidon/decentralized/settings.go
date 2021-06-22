package decentralized

import "go.dedis.ch/kyber/v3/group/edwards25519"

// Suite is the type of keys used to secure communication in Onet
func GetSuit() *edwards25519.SuiteEd25519 {
	return edwards25519.NewBlakeSHA256Ed25519()
}

const approximationDegree uint = 3
const interval float64 = 3
const maxM1N2Ratio float64 = 8.0
const ncells int = 50
const nfilters int = 6
const nmakers int = 48
const nclasses int = 2
const nodeBatchSize int = 5
const learningRate float64 = 0.1
const momentum float64 = 0.9
const maxIterations int = 3

const HOSTS int = 3
