package leastsquares

type leastSquaresKey struct {
	degree uint
	a      uint // interval [-a, a]
}

var leastSquares map[leastSquaresKey][]float64 = map[leastSquaresKey][]float64{
	{3, 1}:   {0.5, 0.2496, 0, -0.0187},
	{3, 3}:   {0.5, 0.6997, 0, -0.2649},
	{3, 5}:   {0.5, 0.9917, 0, -0.5592},
	{3, 7}:   {0.5, 1.1511, 0, -0.7517},
	{5, 1}:   {0.5, 0.25, 0, -0.0207, 0, 0.0018},
	{5, 3}:   {0.5, 0.7392, 0, -0.4489, 0, 0.1656},
	{5, 5}:   {0.5, 1.1466, 0, -1.2824, 0, 0.6509},
	{5, 7}:   {0.5, 1.4290, 0, -2.0487, 0, 1.1674},
	{5, 9}:   {0.5, 1.6117, 0, -2.6069, 0, 1.5680},
	{5, 15}:  {0.5, 1.8624, 0, -3.4375, 0, 2.1913},
	{7, 1}:   {0.6931, 0.5, 0, 0.1250, 0, -0.0052, 0, -0.0003},
	{7, 3}:   {0.5, 0.7478, 0, -0.5267, 0, 0.3368, 0, -0.1059},
	{7, 5}:   {0.5, 1.2109, 0, -1.8606, 0, 1.9228, 0, -0.7874},
	{7, 7}:   {0.5, 1.5856, 0, -3.4583, 0, 4.2684, 0, -1.9197},
	{7, 9}:   {0.5, 1.8619, 0, -4.8585, 0, 6.5216, 0, -3.0665},
	{7, 13}:  {0.5, 2.1991, 0, -6.7689, 0, 9.7850, 0, -4.7879},
	{7, 15}:  {0.5, 2.3004, 0, -7.3792, 0, 10.8630, 0, -5.3682},
	{7, 20}:  {0.6931, 0.5, 0, 0.1250, 0, -0.0052, 0, -0.0003},
	{9, 1}:   {0.5, 0.25, 0, -0.0208, 0, 0.0021, 0, -0.002, 0},
	{9, 3}:   {0.5, 0.7496, 0, -0.5525, 0, 0.4375, 0, -0.2499, 0, 0.0680},
	{9, 5}:   {0.5, 1.2357, 0, -2.2252, 0, 3.3448, 0, -2.8187, 0, 0.9593},
	{9, 7}:   {0.5, 1.6685, 0, -4.6741, 0, 9.0102, 0, -8.6937, 0, 3.1989},
	{9, 9}:   {0.5, 2.0205, 0, -7.1854, 0, 15.5963, 0, -16.0303, 0, 6.1218},
	{9, 13}:  {0.5, 2.4993, 0, -11.1713, 0, 26.9544, 0, -29.3156, 0, 11.5825},
	{9, 15}:  {0.5, 2.6557, 0, -12.5914, 0, 31.1907, 0, -34.4077, 0, 13.7131},
	{9, 20}:  {0.5, 2.9057, 0, -14.9617, 0, 38.4331, 0, -43.2426, 0, 17.4464},
	{9, 25}:  {0.5, 3.0428, 0, -16.3110, -0, 42.6392, 0, -48.4378, -0, 19.6601},
	{9, 30}:  {0.5, 3.1243, 0, -17.1276, -0, 45.2108, 0, -51.6344, -0, 21.0282},
	{13, 15}: {0.5, 3.1517, 0, -25.1341, 0, 122.6275, 0, -321.9625, 0, 455.5079, 0, -327.0613, 0, 93.4012},
	{15, 15}: {0.5, 3.3, 0, -31.6, 0, 196.3, 0, -690.5, 0, 1397.4, 0, -1611.5, 0, 982.6, 0, -245.6},
	{15, 20}: {0.5, 3.9, 0, -44.2, 0, 297.5, 0, -1092.2, 0, 2268.4, 0, -2661.1, 0, 1642.3, 0, -414.1},
}
