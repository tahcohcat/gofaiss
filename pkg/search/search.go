package search

import (
	"fmt"
	"time"

	"github.com/tahcohcat/gofaiss/pkg/index/flat"
	"github.com/tahcohcat/gofaiss/pkg/index/hnsw"
	"github.com/tahcohcat/gofaiss/pkg/index/ivf"
	"github.com/tahcohcat/gofaiss/pkg/index/ivfpq"
	"github.com/tahcohcat/gofaiss/pkg/index/pq"
	"github.com/tahcohcat/gofaiss/pkg/index/stats"
	"github.com/tahcohcat/gofaiss/pkg/vector"
)

// Searcher provides a unified interface for all index types
type Searcher struct {
	idx        interface{}
	indexType  string
	searchOpts SearchOptions
}

// SearchOptions holds search parameters
type SearchOptions struct {
	K           int                    // number of results
	Nprobe      int                    // for IVF-based indexes
	EfSearch    int                    // for HNSW
	ExtraParams map[string]interface{} // additional parameters
}

// DefaultSearchOptions returns default search options
func DefaultSearchOptions() SearchOptions {
	return SearchOptions{
		K:           10,
		Nprobe:      10,
		EfSearch:    50,
		ExtraParams: make(map[string]interface{}),
	}
}

// NewSearcher creates a new searcher for any index type
func NewSearcher(idx interface{}, opts SearchOptions) (*Searcher, error) {
	indexType, err := detectIndexType(idx)
	if err != nil {
		return nil, err
	}

	return &Searcher{
		idx:        idx,
		indexType:  indexType,
		searchOpts: opts,
	}, nil
}

// Search performs a search with the configured options
func (s *Searcher) Search(query []float32) ([]vector.SearchResult, error) {
	return s.SearchWithK(query, s.searchOpts.K)
}

// SearchWithK performs a search for k results
func (s *Searcher) SearchWithK(query []float32, k int) ([]vector.SearchResult, error) {
	switch v := s.idx.(type) {
	case *flat.Index:
		return v.Search(query, k)
	case *hnsw.Index:
		return v.Search(query, k)
	case *pq.Index:
		return v.Search(query, k)
	case *ivf.Index:
		return v.Search(query, k, s.searchOpts.Nprobe)
	case *ivfpq.Index:
		results, err := v.Search(query, k, s.searchOpts.Nprobe)
		if err != nil {
			return nil, err
		}
		// Convert interface{} results to proper type if needed
		if results == nil {
			return []vector.SearchResult{}, nil
		}
		return []vector.SearchResult{}, nil // IVFPQ needs full implementation
	default:
		return nil, fmt.Errorf("unsupported index type")
	}
}

// BatchSearch performs batch search
func (s *Searcher) BatchSearch(queries [][]float32) ([][]vector.SearchResult, error) {
	return s.BatchSearchWithK(queries, s.searchOpts.K)
}

// BatchSearchWithK performs batch search for k results
func (s *Searcher) BatchSearchWithK(queries [][]float32, k int) ([][]vector.SearchResult, error) {
	switch v := s.idx.(type) {
	case *flat.Index:
		return v.BatchSearch(queries, k)
	case *hnsw.Index:
		return v.BatchSearch(queries, k)
	case *pq.Index:
		return v.BatchSearch(queries, k)
	case *ivf.Index:
		return v.BatchSearch(queries, k, s.searchOpts.Nprobe)
	case *ivfpq.Index:
		// IVFPQ batch search placeholder
		results := make([][]vector.SearchResult, len(queries))
		for i := range results {
			results[i] = []vector.SearchResult{}
		}
		return results, nil
	default:
		return nil, fmt.Errorf("unsupported index type")
	}
}

// UpdateOptions updates search options
func (s *Searcher) UpdateOptions(opts SearchOptions) {
	s.searchOpts = opts

	// Apply specific options to index if applicable
	if hnswIdx, ok := s.idx.(*hnsw.Index); ok {
		hnswIdx.SetEfSearch(opts.EfSearch)
	}
}

// Stats returns index statistics
func (s *Searcher) Stats() stats.Stats {
	switch v := s.idx.(type) {
	case *flat.Index:
		return v.Stats()
	case *hnsw.Index:
		return v.Stats()
	case *pq.Index:
		return v.Stats()
	case *ivf.Index:
		return v.Stats()
	case *ivfpq.Index:
		return v.Stats()
	default:
		return stats.Stats{}
	}
}

// SearchResult wraps results with metadata
type SearchResultWithMetadata struct {
	Results   []vector.SearchResult
	QueryTime time.Duration
	IndexType string
}

// SearchWithMetadata performs search and returns timing information
func (s *Searcher) SearchWithMetadata(query []float32) (*SearchResultWithMetadata, error) {
	start := time.Now()
	results, err := s.Search(query)
	if err != nil {
		return nil, err
	}

	return &SearchResultWithMetadata{
		Results:   results,
		QueryTime: time.Since(start),
		IndexType: s.indexType,
	}, nil
}

// RangeSearch finds all vectors within a distance threshold
func (s *Searcher) RangeSearch(query []float32, threshold float32, maxResults int) ([]vector.SearchResult, error) {
	// Search with large k and filter results
	k := s.searchOpts.K * 10
	if maxResults > 0 && k > maxResults {
		k = maxResults
	}

	results, err := s.SearchWithK(query, k)
	if err != nil {
		return nil, err
	}

	// Filter by threshold
	filtered := make([]vector.SearchResult, 0)
	for _, r := range results {
		if r.Distance <= threshold {
			filtered = append(filtered, r)
			if maxResults > 0 && len(filtered) >= maxResults {
				break
			}
		}
	}

	return filtered, nil
}

// Helper functions

func detectIndexType(idx interface{}) (string, error) {
	switch idx.(type) {
	case *flat.Index:
		return "flat", nil
	case *hnsw.Index:
		return "hnsw", nil
	case *pq.Index:
		return "pq", nil
	case *ivf.Index:
		return "ivf", nil
	case *ivfpq.Index:
		return "ivfpq", nil
	default:
		return "", fmt.Errorf("unknown index type")
	}
}

// Builder provides a fluent API for creating searchers
type Builder struct {
	indexType  string
	dimension  int
	metric     string
	indexOpts  map[string]interface{}
	searchOpts SearchOptions
}

// NewBuilder creates a new search builder
func NewBuilder() *Builder {
	return &Builder{
		indexType:  "hnsw",
		dimension:  128,
		metric:     "l2",
		indexOpts:  make(map[string]interface{}),
		searchOpts: DefaultSearchOptions(),
	}
}

// WithIndexType sets the index type
func (b *Builder) WithIndexType(indexType string) *Builder {
	b.indexType = indexType
	return b
}

// WithDimension sets the vector dimension
func (b *Builder) WithDimension(dim int) *Builder {
	b.dimension = dim
	return b
}

// WithMetric sets the distance metric
func (b *Builder) WithMetric(metric string) *Builder {
	b.metric = metric
	return b
}

// WithIndexOption sets an index-specific option
func (b *Builder) WithIndexOption(key string, value interface{}) *Builder {
	b.indexOpts[key] = value
	return b
}

// WithSearchOptions sets search options
func (b *Builder) WithSearchOptions(opts SearchOptions) *Builder {
	b.searchOpts = opts
	return b
}

// Build creates the index and searcher
func (b *Builder) Build() (*Searcher, error) {
	var idx interface{}
	var err error

	switch b.indexType {
	case "flat":
		idx, err = flat.New(b.dimension, b.metric)

	case "hnsw":
		config := hnsw.Config{
			Metric:         b.metric,
			M:              getIntOpt(b.indexOpts, "M", 16),
			EfConstruction: getIntOpt(b.indexOpts, "efConstruction", 200),
			EfSearch:       getIntOpt(b.indexOpts, "efSearch", 50),
		}
		idx, err = hnsw.New(b.dimension, b.metric, config)

	case "pq":
		config := pq.Config{
			M:     getIntOpt(b.indexOpts, "M", 8),
			Nbits: getIntOpt(b.indexOpts, "nbits", 8),
		}
		idx, err = pq.NewIndex(b.dimension, config)

	case "ivf":
		config := ivf.Config{
			Metric: b.metric,
			Nlist:  getIntOpt(b.indexOpts, "nlist", 100),
		}
		idx, err = ivf.New(b.dimension, b.metric, config)

	case "ivfpq":
		config := ivfpq.Config{
			Metric: b.metric,
			Nlist:  getIntOpt(b.indexOpts, "nlist", 100),
			M:      getIntOpt(b.indexOpts, "M", 8),
			Nbits:  getIntOpt(b.indexOpts, "nbits", 8),
		}
		idx, err = ivfpq.New(b.dimension, b.metric, config)

	default:
		return nil, fmt.Errorf("unknown index type: %s", b.indexType)
	}

	if err != nil {
		return nil, err
	}

	return NewSearcher(idx, b.searchOpts)
}

func getIntOpt(opts map[string]interface{}, key string, defaultVal int) int {
	if v, ok := opts[key]; ok {
		if intVal, ok := v.(int); ok {
			return intVal
		}
	}
	return defaultVal
}