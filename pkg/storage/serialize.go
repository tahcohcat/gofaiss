package storage

import (
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"io"
	"os"

	"github.com/tahcohcat/gofaiss/pkg/index/flat"
	"github.com/tahcohcat/gofaiss/pkg/index/hnsw"
	"github.com/tahcohcat/gofaiss/pkg/index/pq"
)

// Serializer handles saving and loading indexes
type Serializer struct {
	compress bool
}

// NewSerializer creates a new serializer
func NewSerializer(compress bool) *Serializer {
	return &Serializer{compress: compress}
}

// SaveFlatIndex saves a flat index to file
func (s *Serializer) SaveFlatIndex(idx *flat.Index, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	var w io.Writer = f
	if s.compress {
		gzw := gzip.NewWriter(f)
		defer gzw.Close()
		w = gzw
	}

	encoder := gob.NewEncoder(w)
	
	// Save index type marker
	if err := encoder.Encode("flat"); err != nil {
		return err
	}
	
	// Save dimension and metric
	if err := encoder.Encode(idx.Dimension()); err != nil {
		return err
	}
	
	// Save vectors
	vectors := idx.GetVectors()
	if err := encoder.Encode(len(vectors)); err != nil {
		return err
	}
	for _, v := range vectors {
		if err := encoder.Encode(v); err != nil {
			return err
		}
	}
	
	return nil
}

// SaveHNSWIndex saves an HNSW index to file
func (s *Serializer) SaveHNSWIndex(idx *hnsw.Index, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	var w io.Writer = f
	if s.compress {
		gzw := gzip.NewWriter(f)
		defer gzw.Close()
		w = gzw
	}

	encoder := gob.NewEncoder(w)
	
	// Save index type marker
	if err := encoder.Encode("hnsw"); err != nil {
		return err
	}
	
	// Note: This is simplified. In practice, you'd need getter methods
	// or make fields public/add serialization methods to hnsw.Index
	stats := idx.Stats()
	if err := encoder.Encode(stats); err != nil {
		return err
	}
	
	return nil
}

// SavePQIndex saves a PQ index to file
func (s *Serializer) SavePQIndex(idx *pq.Index, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	var w io.Writer = f
	if s.compress {
		gzw := gzip.NewWriter(f)
		defer gzw.Close()
		w = gzw
	}

	encoder := gob.NewEncoder(w)
	
	// Save index type marker
	if err := encoder.Encode("pq"); err != nil {
		return err
	}
	
	stats := idx.Stats()
	if err := encoder.Encode(stats); err != nil {
		return err
	}
	
	return nil
}

// LoadIndex loads an index from file and returns the appropriate type
func (s *Serializer) LoadIndex(filename string) (interface{}, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var r io.Reader = f
	if s.compress {
		gzr, err := gzip.NewReader(f)
		if err != nil {
			return nil, err
		}
		defer gzr.Close()
		r = gzr
	}

	decoder := gob.NewDecoder(r)
	
	// Read index type
	var indexType string
	if err := decoder.Decode(&indexType); err != nil {
		return nil, err
	}
	
	switch indexType {
	case "flat":
		return s.loadFlatIndex(decoder)
	case "hnsw":
		return s.loadHNSWIndex(decoder)
	case "pq":
		return s.loadPQIndex(decoder)
	default:
		return nil, fmt.Errorf("unknown index type: %s", indexType)
	}
}

func (s *Serializer) loadFlatIndex(decoder *gob.Decoder) (*flat.Index, error) {
	// Load dimension
	var dim int
	if err := decoder.Decode(&dim); err != nil {
		return nil, err
	}
	
	// Create index (simplified - need metric info)
	idx, err := flat.New(dim, "l2")
	if err != nil {
		return nil, err
	}
	
	// Load vectors
	var count int
	if err := decoder.Decode(&count); err != nil {
		return nil, err
	}
	
	// Note: This would need proper implementation with Vector type
	// vectors := make([]vector.Vector, count)
	// for i := 0; i < count; i++ {
	//     if err := decoder.Decode(&vectors[i]); err != nil {
	//         return nil, err
	//     }
	// }
	// idx.Add(vectors)
	
	return idx, nil
}

func (s *Serializer) loadHNSWIndex(decoder *gob.Decoder) (*hnsw.Index, error) {
	// Simplified implementation
	return nil, fmt.Errorf("HNSW loading not fully implemented")
}

func (s *Serializer) loadPQIndex(decoder *gob.Decoder) (*pq.Index, error) {
	// Simplified implementation
	return nil, fmt.Errorf("PQ loading not fully implemented")
}

// Helper function to save any index type
func SaveIndex(idx interface{}, filename string, compress bool) error {
	s := NewSerializer(compress)
	
	switch v := idx.(type) {
	case *flat.Index:
		return s.SaveFlatIndex(v, filename)
	case *hnsw.Index:
		return s.SaveHNSWIndex(v, filename)
	case *pq.Index:
		return s.SavePQIndex(v, filename)
	default:
		return fmt.Errorf("unsupported index type")
	}
}