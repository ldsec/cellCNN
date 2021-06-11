package utils

import (
	"fmt"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"math"
)

func Histogram(v []float64, filename string) {

	n := int(math.Abs(Max(v) - Min(v)))
	fmt.Printf("\n n = %d", n)
	vals := plotter.Values(v)

	p := plot.New()
	p.Title.Text = "histogram"

	h, err := plotter.NewHist(vals, n)
	if err != nil {
		panic(err)
	}

	p.Add(h)

	if err := p.Save(4*vg.Inch, 4*vg.Inch, filename+".png"); err != nil {
		panic(err)
	}

}
