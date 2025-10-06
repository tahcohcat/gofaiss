package pq

import "fmt"

// Config is placeholder PQ config
type Config struct {
	M     int
	Nbits int
}

// Index is a placeholder PQ index
type Index struct {
	dim int
}

func NewIndex(dim int, cfg Config) (*Index, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("invalid dim")
	}
	return &Index{dim: dim}, nil
}

// Train placeholder
func (idx *Index) Train(vs any) error { return nil }

// Add placeholder
func (idx *Index) Add(vs any) error { return nil }

// Search placeholder
func (idx *Index) Search(q []float32, k int) (any, error) { return nil, nil }

func (idx *Index) Stats() any { return nil }
