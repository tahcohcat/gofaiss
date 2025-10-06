package stats

type Stats struct {
	TotalVectors  int
	Dimension     int
	IndexType     string
	MemoryUsageMB float64
	ExtraInfo     map[string]any
}
