package utils

// return the size of a two dim slice
func SizeOf2DimSlice(data interface{}) int {
	size := 0
	switch data := data.(type) {
	case [][]byte:
		for i := range data {
			size += len(data[i])
		}
	case [][]float64:
		for i := range data {
			size += len(data[i])
		}
	}
	return size
}
