package vector

// Vector is the public vector type used across packages
// Vector represents a single vector with an ID
type Vector struct {
	ID   int64
	Data []float32
	Norm float32
}
