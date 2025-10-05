package index

// Index defines the public indexing interface used across implementations.

import (
	"github.com/tahcohcat/gofaiss/pkg/search"
	"github.com/tahcohcat/gofaiss/pkg/vector"
)

// Index interface defines the basic operations for vector indexing
type Index interface {
	Add(vectors []vector.Vector) error
	Search(query []float32, k int) ([]search.Result, error)
	Remove(id int64) error
	Size() int
}
