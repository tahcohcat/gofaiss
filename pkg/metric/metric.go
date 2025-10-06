package metric

import "fmt"

// Type is a metric type
type Type string

const (
	L2     Type = "l2"
	Cosine Type = "cosine"
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
	default:
		return nil, fmt.Errorf("unknown metric %s", t)
	}
}

type l2 struct{}

func (l l2) Distance(a, b []float32) float32 { return 0 }
func (l l2) Name() string                    { return "l2" }

type cosine struct{}

func (c cosine) Distance(a, b []float32) float32 { return 0 }
func (c cosine) Name() string                    { return "cosine" }
