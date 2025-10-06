package metric

import (
	"fmt"

	internalmath "github.com/tahcohcat/gofaiss/internal/math"
)

// Type is a metric type
type Type string

const (
	L2     Type = "l2"
	Cosine Type = "cosine"
	Dot    Type = "dot"
)

// Metric is an interface for distance computations
type Metric interface {
	Distance(a, b []float32) float32
	Name() string
}

// New returns a metric by type
func New(t Type) (Metric, error) {
	switch t {
	case L2:
		return l2{}, nil
	case Cosine:
		return cosine{}, nil
	case Dot:
		return dot{}, nil
	default:
		return nil, fmt.Errorf("unknown metric %s", t)
	}
}

type l2 struct{}

func (l l2) Distance(a, b []float32) float32 {
	return internalmath.L2Distance(a, b)
}

func (l l2) Name() string {
	return "l2"
}

type cosine struct{}

func (c cosine) Distance(a, b []float32) float32 {
	return internalmath.CosineDistance(a, b)
}

func (c cosine) Name() string {
	return "cosine"
}

type dot struct{}

func (d dot) Distance(a, b []float32) float32 {
	return internalmath.InnerProduct(a, b)
}

func (d dot) Name() string {
	return "dot"
}