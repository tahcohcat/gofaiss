package vector

import (
	"fmt"
	"math"
	"math/rand"
)

// Vector represents an ID + data
type Vector struct {
	ID   int64
	Data []float32
	Norm float32 // Precomputed norm for cosine similarity
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

// Add adds two vectors element-wise
func Add(a, b []float32) []float32 {
	result := make([]float32, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result
}

// Subtract subtracts b from a element-wise
func Subtract(a, b []float32) []float32 {
	result := make([]float32, len(a))
	for i := range a {
		result[i] = a[i] - b[i]
	}
	return result
}

// Scale multiplies vector by scalar
func Scale(v []float32, s float32) []float32 {
	result := make([]float32, len(v))
	for i := range v {
		result[i] = v[i] * s
	}
	return result
}

// Norm computes L2 norm (magnitude) of a vector
func Norm(v []float32) float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	return float32(math.Sqrt(float64(sum)))
}

// Normalize normalizes vector to unit length
func Normalize(v []float32) []float32 {
	norm := Norm(v)
	if norm == 0 {
		return v
	}
	return Scale(v, 1.0/norm)
}

// NormalizeInPlace normalizes vector in-place
func NormalizeInPlace(v []float32) {
	norm := Norm(v)
	if norm == 0 {
		return
	}
	scale := 1.0 / norm
	for i := range v {
		v[i] *= scale
	}
}

// Centroid computes the mean of a set of vectors
func Centroid(vectors [][]float32) []float32 {
	if len(vectors) == 0 {
		return nil
	}

	dim := len(vectors[0])
	result := make([]float32, dim)

	for _, v := range vectors {
		for i := range v {
			result[i] += v[i]
		}
	}

	scale := 1.0 / float32(len(vectors))
	for i := range result {
		result[i] *= scale
	}

	return result
}
