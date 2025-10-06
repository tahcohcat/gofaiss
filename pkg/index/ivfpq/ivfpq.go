package ivfpq

import "fmt"

// Config placeholder for IVFPQ
type Config struct {
	Metric string
	Nlist  int
	M      int
	Nbits  int
}

// Index placeholder
type Index struct {
	dim int
}

func New(dim int, metric string, cfg Config) (*Index, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("invalid dim")
	}
	return &Index{dim: dim}, nil
}

func (idx *Index) Train(vs any) error                                   { return nil }
func (idx *Index) Add(vs any) error                                     { return nil }
func (idx *Index) Search(q []float32, k int, nprobe int) ([]any, error) { return nil, nil }
func (idx *Index) Stats() any                                           { return nil }
