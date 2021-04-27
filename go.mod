module github.com/ldsec/cellCNN

go 1.14

replace github.com/ldsec/lattigo/v2 => C:\Users\Zybeline\go\src\github.com\ldsec\lattigo

require (
	github.com/daviddengcn/go-colortext v1.0.0 // indirect
	github.com/gorilla/websocket v1.4.2 // indirect
	github.com/ldsec/lattigo/v2 v2.1.2-0.20210417094739-e3ef9087b323
	github.com/ldsec/unlynx v1.4.1
	github.com/montanaflynn/stats v0.6.3
	github.com/pelletier/go-toml v1.9.0 // indirect
	github.com/stretchr/testify v1.7.0
	go.dedis.ch/kyber/v3 v3.0.12
	go.dedis.ch/onet/v3 v3.2.3
	gonum.org/v1/gonum v0.8.1
	gonum.org/v1/plot v0.8.0
)

//replace go.dedis.ch/onet/v3 => ../../../go.dedis.ch/onet

//replace github.com/ldsec/unlynx => ../unlynx
