package utils

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

func ComputeAccuracy(c []float64, y []float64) float64{
	accuracy := 0.
	for i:= range y {
		if c[i] == y[i] {
			accuracy++
		}
	}
	accuracy = 100 * accuracy / float64(len(y))
	return accuracy
}

func ComputePrecisionRecall(c []float64, y []float64) (float64, float64){
	tp := 0.
	fp := 0.
	fn := 0.
	for i:= range y {
		if c[i] >= 1 {
			if y[i] >= 1 {
				tp++
			}else{
				fp++
			}
		}else{
			if y[i] >= 1 {
				fn++
			}
		}
	}
	precision := 100 * tp/(fp + tp)
	recall := 100 * tp/(fn + tp)
	return precision, recall
}

func classifyDTI(x float64) float64{
	if x < 0 {
		return -1
	}
	return 1
}

func ClassifyCellCNN(scores *mat.Dense) []float64 {
	nsamples, _ := scores.Dims()
	class := make([]float64, nsamples)
	for r := range class {
		if scores.At(r, 0) < scores.At(r, 1) {
			class[r] = 1
		} else {
			class[r] = 0
		}
	}
	return class
}

func Print_train_stats_dti(scores []float64, y []float64) []float64 {

	classified := Apply(classifyDTI, scores)
	accuracy := ComputeAccuracy(classified, y)

	fmt.Println("\nscores : ", scores)
	fmt.Println("\nclassified: ", classified)
	fmt.Println("\ny: ", y)

	fmt.Printf("\nAccuracy: %.2f %% \n ---\n\n", accuracy)

	return classified
}

func Print_train_stats_cellCNN(scores *mat.Dense, y []float64){
	classified := ClassifyCellCNN(scores)
	accuracy := ComputeAccuracy(classified, y)
	precision, recall := ComputePrecisionRecall(classified, y)

	//fmt.Println("scores : ", scores)
	//fmt.Println("classified: ", classified)
	//fmt.Println("y: ", y)
	fmt.Printf("Accuracy: %.2f %%, precision: %.2f %%, recall: %.2f %% \n\n", accuracy, precision, recall)


}