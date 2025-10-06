package vector

import (
	"fmt"
	"math/rand"
)

// Vector represents an ID + data
type Vector struct {
	ID   int64
	Data []float32
}

// SearchResult holds a neighbor
type SearchResult struct {
	ID       int64
	Distance float32
}

// GenerateRandom produces n random vectors of dimension dim
func GenerateRandom(n, dim int, seed int64) []Vector {
	r := rand.New(rand.NewSource(seed))
	res := make([]Vector, n)
	for i := 0; i < n; i++ {
		data := make([]float32, dim)
		for j := 0; j < dim; j++ {
			data[j] = float32(r.Float32())
		}
		res[i] = Vector{ID: int64(i), Data: data}
	}
	return res
}

// ValidateDimension ensures each vector has the expected dimension
func ValidateDimension(vs []Vector, dim int) error {
	for _, v := range vs {
		if len(v.Data) != dim {
			return fmt.Errorf("vector dim mismatch")
		}
	}
	return nil
}

// Copy returns a copy of a slice
func Copy(s []float32) []float32 {
	c := make([]float32, len(s))
	copy(c, s)
	return c
}
