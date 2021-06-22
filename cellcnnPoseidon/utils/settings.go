package utils

// CellCnnSettings store the cell cnn basic settings
type CellCnnSettings struct {
	Ncells   int
	Nmakers  int
	Nfilters int
	Nclasses int
	Degree   uint
	Interval float64
}

// NewCellCnnSettings constructor
func NewCellCnnSettings(
	ncells, nmakers, nfilters, nclasses int, degree uint, interval float64,
) *CellCnnSettings {

	return &CellCnnSettings{
		Ncells:   ncells,
		Nmakers:  nmakers,
		Nfilters: nfilters,
		Nclasses: nclasses,
		Degree:   degree,
		Interval: interval,
	}
}
