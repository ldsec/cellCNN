package utils

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

func ComputeAccuracy(c []float64, y []float64) float64 {

	accuracy := 0.
	for i := range y {
		if c[i] == y[i] {
			accuracy++
		}
	}
	accuracy = 100 * accuracy / float64(len(y))
	return accuracy
}

func ComputePrecisionRecall(c []float64, y []float64, numClass int, micro bool) (float64, float64) {
	precision, recall := 0.0, 0.0
	if numClass == 2 {
		tp := 0.
		fp := 0.
		fn := 0.
		for i := range y {
			if c[i] >= 1 {
				if y[i] >= 1 {
					tp++
				} else {
					fp++
				}
			} else {
				if y[i] >= 1 {
					fn++
				}
			}
		}
		precision = 100 * tp / (fp + tp)
		recall = 100 * tp / (fn + tp)
	} else {
		c0tp, c0fp, c0fn := 0., 0., 0.
		c1tp, c1fp, c1fn := 0., 0., 0.
		c2tp, c2fp, c2fn := 0., 0., 0.
		for i := range y {
			if y[i] == 0 {
				if c[i] == 0 {
					c0tp++
				}
				if c[i] == 1 {
					c0fn++
					c1fp++
				}
				if c[i] == 2 {
					c0fn++
					c2fp++
				}
			}
			if y[i] == 1 {
				if c[i] == 1 {
					c1tp++
				}
				if c[i] == 0 {
					c1fn++
					c0fp++
				}
				if c[i] == 2 {
					c1fn++
					c2fp++
				}
			}
			if y[i] == 2 {
				if c[i] == 2 {
					c2tp++
				}
				if c[i] == 1 {
					c2fn++
					c1fp++
				}
				if c[i] == 0 {
					c2fn++
					c0fp++
				}
			}
		}
		//class-based precision recalls
		c0p := 100 * c0tp / (c0fp + c0tp)
		c0r := 100 * c0tp / (c0fn + c0tp)
		c1p := 100 * c1tp / (c1fp + c1tp)
		c1r := 100 * c1tp / (c1fn + c1tp)
		c2p := 100 * c2tp / (c2fp + c2tp)
		c2r := 100 * c2tp / (c2fn + c2tp)
		//micro average
		if micro {
			precision = 100 * (c0tp + c1tp + c2tp) / (c0fp + c1fp + c2fp + c0tp + c1tp + c2tp)
			recall = 100 * (c0tp + c1tp + c2tp) / (c0fn + c1fn + c2fn + c0tp + c1tp + c2tp)

		} else {
			//macro average
			precision = (c0p + c1p + c2p) / 3
			recall = (c0r + c1r + c2r) / 3

		}

	}

	return precision, recall
}

func classifyDTI(x float64) float64 {
	if x < 0 {
		return -1
	}
	return 1
}

func ClassifyCellCNN(scores *mat.Dense, nclass int) []float64 {
	nsamples, _ := scores.Dims()
	class := make([]float64, nsamples)
	if nclass == 2 {

		for r := range class {
			if scores.At(r, 0) < scores.At(r, 1) {
				class[r] = 1
			} else {
				class[r] = 0
			}
		}
		return class
	}
	if nclass == 3 {
		for r := range class {
			if scores.At(r, 0) < scores.At(r, 1) && scores.At(r, 2) < scores.At(r, 1) {
				class[r] = 1
			}
			if scores.At(r, 1) < scores.At(r, 0) && scores.At(r, 2) < scores.At(r, 0) {
				class[r] = 0
			}
			if scores.At(r, 1) < scores.At(r, 2) && scores.At(r, 0) < scores.At(r, 2) {
				class[r] = 2
			}
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

func Print_train_stats_cellCNN(scores *mat.Dense, y []float64, nclass int, micro bool) {
	classified := ClassifyCellCNN(scores, nclass)
	accuracy := ComputeAccuracy(classified, y)
	precision, recall := ComputePrecisionRecall(classified, y, nclass, micro)

	//fmt.Println("scores : ", scores)
	//fmt.Println("classified: ", classified)
	//fmt.Println("y: ", y)
	fmt.Printf("Accuracy: %.2f %%, precision: %.2f %%, recall: %.2f %% \n\n", accuracy, precision, recall)

}
