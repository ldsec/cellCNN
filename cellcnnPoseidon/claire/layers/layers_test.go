package layers

/*
import (
	"fmt"
	"github.com/ldsec/cellCNN/semester_project_claire/utils"
	"gonum.org/v1/gonum/mat"
	"os"
	"testing"
)

const h = 3
const n = 2
const m = 4

func TestOutput(t *testing.T) {
	W := []float64{1, 2, 3}  // 1 x h
	b := 2.5
	Z := []float64{2, 2, 1, 2, 4, 2, 4, 4, 6, 5, 4, 6}  // h x m
	y := []float64{0, 1, 1, 0}
	output_layer := Output{Weights: mat.NewDense(1, h, W), Bias: b, Activation:utils.Id}
	scores := output_layer.Forward(mat.NewDense(h, m, Z))
	if len(scores) != m {
		t.Errorf("scores != m = %d but is %d", m, len(scores))
	}
	fmt.Print("scores: ")
	fmt.Println(scores)

	next_delta := output_layer.Backward(scores, y, 0.1, 0.7, 0.2)
	fmt.Print("next delta: ")
	fmt.Println(next_delta)
}

func TestWrite(t *testing.T) {
	file, _ := os.Create("data/example.txt")
	file.WriteString("hello\n")
	file.WriteString("world")
	defer file.Close()
}


func TestConv1D(t *testing.T) {

	ncells := 3
	nmarkers := 5
	nfilters := 4
	input := mat.NewDense(ncells, nmarkers, utils.FillNorm(ncells*nmarkers, 0, 5))
	conv := Conv1D{Nfilters: nfilters}
	output := conv.Forward(input)

	fmt.Println("forward prop output:")
	fmt.Println(output)
	fmt.Println()

	fmt.Println("filters before backprop:")
	fmt.Println(conv.getFilters())

	conv.Backward(mat.NewDense(ncells, nfilters, utils.FillNorm(ncells*nfilters, 0, 5)), 0.3)

	fmt.Println()
	fmt.Println("filters after backprop")
	fmt.Println(conv.getFilters())

}
*/
