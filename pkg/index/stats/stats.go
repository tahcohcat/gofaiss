package stats

// Stats holds index statistics
type Stats struct {
	TotalVectors  int
	Dimension     int
	IndexType     string
	MemoryUsageMB float64
	ExtraInfo     map[string]interface{}
}

// Empty returns an empty stats object
func Empty() Stats {
	return Stats{
		ExtraInfo: make(map[string]interface{}),
	}
}

// NewStats creates a new Stats object with basic info
func NewStats(totalVectors, dimension int, indexType string, memoryMB float64) Stats {
	return Stats{
		TotalVectors:  totalVectors,
		Dimension:     dimension,
		IndexType:     indexType,
		MemoryUsageMB: memoryMB,
		ExtraInfo:     make(map[string]interface{}),
	}
}

// WithExtra adds extra information to stats
func (s Stats) WithExtra(key string, value interface{}) Stats {
	if s.ExtraInfo == nil {
		s.ExtraInfo = make(map[string]interface{})
	}
	s.ExtraInfo[key] = value
	return s
}
