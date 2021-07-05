module github.com/ldsec/cellCNN

go 1.13

require (
	github.com/ldsec/lattigo/v2 v2.1.2-0.20210611151535-f43eec625b04
	github.com/stretchr/testify v1.7.0
	go.dedis.ch/kyber/v3 v3.0.13
	go.dedis.ch/onet/v3 v3.2.9
	gonum.org/v1/gonum v0.9.3
	gonum.org/v1/plot v0.9.0
)

//replace go.dedis.ch/onet/v3 => ../../../go.dedis.ch/onet
