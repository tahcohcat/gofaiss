package flat

// Flat (brute-force) index implementation placeholder.
import (
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"os"
	"sort"
	"sync"
	"github.com/tahcohcat/gofaiss/pkg/vector"
	"github.com/tahcohcat/gofaiss/pkg/search"
)


func New(dim int) *FlatIndex {
	return &FlatIndex{}
}
// FlatIndex implements a simple flat (brute-force) index
type FlatIndex struct {
	dim     int
	metric  string
	vectors []vector.Vector
	mu      sync.RWMutex
}

func NewFlatIndex(dim int, metric string) (*FlatIndex, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	if metric != "l2" && metric != "cosine" {
		return nil, fmt.Errorf("metric must be 'l2' or 'cosine'")
	}
	return &FlatIndex{
		dim:     dim,
		metric:  metric,
		vectors: make([]vector.Vector, 0),
	}, nil
}

func (idx *FlatIndex) Add(vectors []vector.Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for _, v := range vectors {
		if len(v.Data) != idx.dim {
			return fmt.Errorf("vector dimension mismatch: expected %d, got %d", idx.dim, len(v.Data))
		}
		if idx.metric == "cosine" {
			v.Norm = computeNorm(v.Data)
			if v.Norm == 0 {
				return fmt.Errorf("zero vector not allowed for cosine metric")
			}
		}
		idx.vectors = append(idx.vectors, v)
	}
	return nil
}

func (idx *FlatIndex) Search(query []float32, k int) ([]search.Result, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(query) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", idx.dim, len(query))
	}

	if k <= 0 || k > len(idx.vectors) {
		k = len(idx.vectors)
	}

	queryNorm := float32(0)
	if idx.metric == "cosine" {
		queryNorm = computeNorm(query)
		if queryNorm == 0 {
			return nil, fmt.Errorf("zero query vector not allowed for cosine metric")
		}
	}

	results := make([]SearchResult, len(idx.vectors))
	for i, v := range idx.vectors {
		var dist float32
		if idx.metric == "l2" {
			dist = l2Distance(query, v.Data)
		} else {
			dist = cosineDistance(query, v.Data, queryNorm, v.Norm)
		}
		results[i] = SearchResult{ID: v.ID, Distance: dist}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	if k < len(results) {
		return results[:k], nil
	}
	return results, nil
}

func (idx *FlatIndex) LoadFromFile(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	gzr, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gzr.Close()

	idx.mu.Lock()
	defer idx.mu.Unlock()

	decoder := gob.NewDecoder(gzr)
	if err := decoder.Decode(&idx.dim); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.metric); err != nil {
		return err
	}
	var count int
	if err := decoder.Decode(&count); err != nil {
		return err
	}
	idx.vectors = make([]Vector, count)
	for i := 0; i < count; i++ {
		if err := decoder.Decode(&idx.vectors[i]); err != nil {
			return err
		}
	}
	return nil
}