package search

// Unified search API that wraps index implementations.


type Result struct {
	ID       int64
	Distance float32
}
