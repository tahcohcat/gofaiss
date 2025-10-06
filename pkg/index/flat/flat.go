package flat

import (
	"fmt"
	"sort"
	"sync"

	internalmath "github.com/tahcohcat/gofaiss/internal/math"
	"github.com/tahcohcat/gofaiss/pkg/index/stats"
	"github.com/tahcohcat/gofaiss/pkg/storage"
	"github.com/tahcohcat/gofaiss/pkg/vector"
)

// Index is a simple flat (brute-force) index placeholder
type Index struct {
	dim int
	metric  string // "l2" or "cosine"
	vectors []vector.Vector
	mu      sync.RWMutex
}

// New creates a new flat index
func New(dim int, metric string) (*Index, error) {
		if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	if metric != "l2" && metric != "cosine" {
		return nil, fmt.Errorf("metric must be 'l2' or 'cosine'")
	}
	return &Index{
		dim:     dim,
		metric:  metric,
		vectors: make([]vector.Vector, 0),
	}, nil

}

// Add adds vectors to the index (no-op placeholder)
func (idx *Index) Add(vs []vector.Vector) error { 	
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for _, v := range vs {
		if len(v.Data) != idx.dim {
			return fmt.Errorf("vector dimension mismatch: expected %d, got %d", idx.dim, len(v.Data))
		}
		
		// Compute norm for cosine similarity
		if idx.metric == "cosine" {
			v.Norm = internalmath.Norm(v.Data)
			if v.Norm == 0.0 {
				return fmt.Errorf("zero vector not allowed for cosine metric")
			}
		}
		
		idx.vectors = append(idx.vectors, v)
	}
	return nil 
}

// BatchSearch performs batch search
func (idx *Index) BatchSearch(queries [][]float32, k int) ([][]vector.SearchResult, error) {
	results := make([][]vector.SearchResult, len(queries))
	for i, query := range queries {
		res, err := idx.Search(query, k)
		if err != nil {
			return nil, err
		}
		results[i] = res
	}
	return results, nil
}

// Search returns empty results placeholder
func (idx *Index) Search(q []float32, k int) ([]vector.SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(q) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", idx.dim, len(q))
	}

	if k <= 0 || k > len(idx.vectors) {
		k = len(idx.vectors)
	}

	queryNorm := float32(0)
	if idx.metric == "cosine" {
		queryNorm = internalmath.Norm(q)
		if queryNorm == 0 {
			return nil, fmt.Errorf("zero query vector not allowed for cosine metric")
		}
	}

	results := make([]vector.SearchResult, len(idx.vectors))
	for i, v := range idx.vectors {
		var dist float32
		if idx.metric == "l2" {
			dist = internalmath.L2Distance(q, v.Data)
		} else {
			dist = internalmath.CosineDistanceWithNorms(q, v.Data, queryNorm, v.Norm)
		}
		results[i] = vector.SearchResult{ID: v.ID, Distance: dist}
	}

	// Sort by distance (ascending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	if k < len(results) {
		return results[:k], nil
	}
	return results, nil
}

func (idx *Index) Dimension() int { return idx.dim }

func (idx *Index) GetVectors() []vector.Vector {
	return idx.vectors
}

// In pkg/index/pq/pq.go
func (idx *Index) Save(w storage.Writer) error {
    idx.mu.Lock()
    defer idx.mu.Unlock()
    
    if err := w.Encode(idx.dim); err != nil {
        return err
    }
    if err := w.Encode(idx.vectors); err != nil {
        return err
    }
    return nil
}

func (idx *Index) Load(r storage.Reader) error {
    idx.mu.Lock()
    defer idx.mu.Unlock()
    
    if err := r.Decode(&idx.dim); err != nil {
        return err
    }
    if err := r.Decode(&idx.vectors); err != nil {
        return err
    }
    return nil
}


// Stats returns minimal stats
// todo: implement stats
func (idx *Index) Stats() stats.Stats { return stats.Stats{} }
