package centralized

import (
	"testing"
)

func TestCellCnn(t *testing.T) {
	nepochs := 5
	cellCNN(nepochs, false)
}

//func TestTime (t *testing.T){
//	start := time.Now()
//	cellCNN(1, true)
//	timing := time.Since(start)
//	fmt.Printf("Total time: %s\n", timing)
//}
