module github.com/ldsec/cellCNN

go 1.13

require (
	github.com/BurntSushi/toml v0.3.1
	github.com/ldsec/lattigo/v2 v2.1.2-0.20210611151535-f43eec625b04
	github.com/ldsec/unlynx v1.4.1
	github.com/stretchr/testify v1.7.0
	go.dedis.ch/kyber/v3 v3.0.13
	go.dedis.ch/onet/v3 v3.2.9
	gonum.org/v1/gonum v0.9.2
	gonum.org/v1/plot v0.9.0
)

replace go.dedis.ch/onet/v3 => ../onet
