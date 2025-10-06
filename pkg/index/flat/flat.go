package flat

import (
	"fmt"

	"github.com/tahcohcat/gofaiss/pkg/vector"
)

// Index is a simple flat (brute-force) index placeholder
type Index struct {
	dim int
}

// New creates a new flat index
func New(dim int, metric string) (*Index, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("invalid dim")
	}
	return &Index{dim: dim}, nil
}

// Add adds vectors to the index (no-op placeholder)
func (idx *Index) Add(vs []vector.Vector) error { return nil }

// Search returns empty results placeholder
func (idx *Index) Search(q []float32, k int) ([]vector.SearchResult, error) {
	return []vector.SearchResult{}, nil
}

// Stats returns minimal stats
func (idx *Index) Stats() interface{} { return nil }
