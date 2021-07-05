package cellCNN

func (c *CellCNNProtocol) ForwardPlain(XBatch *Matrix) {
	// Convolution
	c.P.MulMat(XBatch, c.C)

	// Dense
	c.U.MulMat(c.P, c.W)
}

func (c *CellCNNProtocol) BackWardPlain(XBatch, YBatch *Matrix, nParties int) {

	L1Batch := new(Matrix)
	L1DerivBatch := new(Matrix)
	E0Batch := new(Matrix)
	E1Batch := new(Matrix)

	// Activations
	L1Batch.Func(c.U, Activation)
	L1DerivBatch.Func(c.U, ActivationDeriv)

	// Dense error
	E1Batch.Sub(L1Batch, YBatch)
	E1Batch.Dot(E1Batch, L1DerivBatch)

	// Convolution error
	E0Batch.MulMat(E1Batch, c.W.Transpose())

	// Updated weights
	c.DW.MulMat(c.P.Transpose(), E1Batch)
	c.DC.MulMat(XBatch.Transpose(), E0Batch)

	c.DW.MultConst(c.DW, complex(LearningRate/float64(nParties), 0))
	c.DC.MultConst(c.DC, complex(LearningRate/float64(nParties), 0))

	if c.DCPrev == nil {
		c.DCPrev = NewMatrix(Features, Filters)
	}

	if c.DWPrev == nil {
		c.DWPrev = NewMatrix(Filters, Classes)
	}

	// Adds the previous weights
	// W_i = learning_rate * Wt + W_i-1 * momentum
}

func (c *CellCNNProtocol) UpdatePlain() {

	// W_i = learning_rate * Wt + W_i-1 * momentum
	c.DC.Add(c.DC, c.DCPrev)
	c.DW.Add(c.DW, c.DWPrev)

	// W_i * momentum
	c.DCPrev.MultConst(c.DC, complex(Momentum, 0))
	c.DWPrev.MultConst(c.DW, complex(Momentum, 0))

	// W = W - W_i
	c.C.Sub(c.C, c.DC)
	c.W.Sub(c.W, c.DW)
}

func (c *CellCNNProtocol) PredictPlain(XBatch *Matrix) *Matrix {

	// Convolution
	c.P.MulMat(XBatch, c.C)

	U := new(Matrix)

	// Dense
	U.MulMat(c.P, c.W)

	// Activations
	U.Func(U, Activation)

	return U
}