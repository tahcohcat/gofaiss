package hnsw

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"

	"github.com/tahcohcat/gofaiss/pkg/index/stats"
	"github.com/tahcohcat/gofaiss/pkg/metric"
	"github.com/tahcohcat/gofaiss/pkg/vector"
)

// Index implements HNSW (Hierarchical Navigable Small World) index
type Index struct {
	dim            int
	metric         metric.Metric
	M              int
	efConstruction int
	efSearch       int
	maxLevel       int
	entryPoint     int64
	nodes          map[int64]*Node
	levelMult      float64
	mu             sync.RWMutex
	nextID         int64
}

// Config holds HNSW configuration
type Config struct {
	Metric         string
	M              int // number of connections per layer
	EfConstruction int // size of dynamic candidate list during construction
	EfSearch       int // size of dynamic candidate list during search (can be adjusted)
}

// DefaultConfig returns default HNSW configuration
func DefaultConfig() Config {
	return Config{
		Metric:         "l2",
		M:              16,
		EfConstruction: 200,
		EfSearch:       200,
	}
}

// New creates a new HNSW index
func New(dim int, metricType string, config Config) (*Index, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	if config.M <= 0 {
		config.M = 16
	}
	if config.EfConstruction <= 0 {
		config.EfConstruction = 200
	}
	if config.EfSearch <= 0 {
		config.EfSearch = config.EfConstruction
	}

	m, err := metric.New(metric.Type(metricType))
	if err != nil {
		return nil, err
	}

	return &Index{
		dim:            dim,
		metric:         m,
		M:              config.M,
		efConstruction: config.EfConstruction,
		efSearch:       config.EfSearch,
		maxLevel:       0,
		entryPoint:     -1,
		nodes:          make(map[int64]*Node),
		levelMult:      1.0 / math.Log(float64(config.M)),
		nextID:         0,
	}, nil
}

// SetEfSearch adjusts the search-time ef parameter
func (idx *Index) SetEfSearch(ef int) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.efSearch = ef
}

// Add adds vectors to the index
func (idx *Index) Add(vectors []vector.Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if err := vector.ValidateDimension(vectors, idx.dim); err != nil {
		return err
	}

	for _, v := range vectors {
		id := v.ID
		if id == 0 {
			id = idx.nextID
			idx.nextID++
		}

		level := idx.randomLevel()
		if level > idx.maxLevel {
			idx.maxLevel = level
		}

		node := &Node{
			ID:    id,
			Data:  v.Data,
			Level: level,
			Edges: make([][]int64, level+1),
		}

		for i := 0; i <= level; i++ {
			node.Edges[i] = make([]int64, 0)
		}

		if idx.entryPoint == -1 {
			idx.entryPoint = id
			idx.nodes[id] = node
			continue
		}

		idx.insertNode(node)
		idx.nodes[id] = node
	}

	return nil
}

// Search finds k nearest neighbors
func (idx *Index) Search(query []float32, k int) ([]vector.SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(query) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch")
	}

	if idx.entryPoint == -1 {
		return []vector.SearchResult{}, nil
	}

	curr := idx.entryPoint
	currDist := idx.metric.Distance(query, idx.nodes[curr].Data)

	// Greedy search through upper layers
	for lc := idx.maxLevel; lc > 0; lc-- {
		changed := true
		for changed {
			changed = false
			node := idx.nodes[curr]
			if lc < len(node.Edges) {
				for _, neighborID := range node.Edges[lc] {
					d := idx.metric.Distance(query, idx.nodes[neighborID].Data)
					if d < currDist {
						currDist = d
						curr = neighborID
						changed = true
					}
				}
			}
		}
	}

	// Search at layer 0
	candidates := idx.searchLayer(query, curr, idx.efSearch, 0)

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Distance < candidates[j].Distance
	})

	if k < len(candidates) {
		return candidates[:k], nil
	}
	return candidates, nil
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

// Remove removes a node
func (idx *Index) Remove(id int64) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	node, exists := idx.nodes[id]
	if !exists {
		return fmt.Errorf("node %d not found", id)
	}

	// Remove edges pointing to this node
	for _, otherNode := range idx.nodes {
		for lc := 0; lc < len(otherNode.Edges); lc++ {
			filtered := make([]int64, 0)
			for _, nid := range otherNode.Edges[lc] {
				if nid != id {
					filtered = append(filtered, nid)
				}
			}
			otherNode.Edges[lc] = filtered
		}
	}

	// Update entry point if needed
	if idx.entryPoint == id {
		if len(idx.nodes) > 1 {
			for _, n := range idx.nodes {
				if n.ID != id && n.Level == idx.maxLevel {
					idx.entryPoint = n.ID
					break
				}
			}
		} else {
			idx.entryPoint = -1
			idx.maxLevel = 0
		}
	}

	delete(idx.nodes, id)
	return nil
}

// Size returns number of nodes
func (idx *Index) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.nodes)
}

// Dimension returns vector dimension
func (idx *Index) Dimension() int {
	return idx.dim
}

// Stats returns index statistics
func (idx *Index) Stats() index.Stats {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	vectorMem := len(idx.nodes) * idx.dim * 4
	edgeMem := len(idx.nodes) * idx.M * 8 * 4
	memoryMB := float64(vectorMem+edgeMem) / (1024 * 1024)

	return index.Stats{
		TotalVectors:  len(idx.nodes),
		Dimension:     idx.dim,
		IndexType:     "HNSW",
		MemoryUsageMB: memoryMB,
		ExtraInfo: map[string]interface{}{
			"metric":         idx.metric.Name(),
			"M":              idx.M,
			"efConstruction": idx.efConstruction,
			"efSearch":       idx.efSearch,
			"maxLevel":       idx.maxLevel,
		},
	}
}

// Internal methods

func (idx *Index) randomLevel() int {
	r := -math.Log(1.0 - float64(rand.Intn(1000000))/1000000.0)
	level := int(r * idx.levelMult)
	if level > 16 {
		level = 16
	}
	return level
}

func (idx *Index) insertNode(node *Node) {
	curr := idx.entryPoint
	currDist := idx.metric.Distance(node.Data, idx.nodes[curr].Data)

	for lc := idx.maxLevel; lc > node.Level; lc-- {
		changed := true
		for changed {
			changed = false
			for _, neighborID := range idx.nodes[curr].Edges[lc] {
				d := idx.metric.Distance(node.Data, idx.nodes[neighborID].Data)
				if d < currDist {
					currDist = d
					curr = neighborID
					changed = true
				}
			}
		}
	}

	for lc := node.Level; lc >= 0; lc-- {
		candidates := idx.searchLayer(node.Data, curr, idx.efConstruction, lc)

		M := idx.M
		if lc == 0 {
			M *= 2
		}

		neighbors := idx.selectNeighbors(candidates, M)

		for _, neighborID := range neighbors {
			node.Edges[lc] = append(node.Edges[lc], neighborID)
			idx.nodes[neighborID].Edges[lc] = append(idx.nodes[neighborID].Edges[lc], node.ID)

			if len(idx.nodes[neighborID].Edges[lc]) > M {
				idx.pruneConnections(neighborID, lc, M)
			}
		}

		if len(candidates) > 0 {
			curr = candidates[0].ID
		}
	}
}

func (idx *Index) searchLayer(query []float32, entryPoint int64, ef int, layer int) []vector.SearchResult {
	visited := make(map[int64]bool)
	candidates := make([]vector.SearchResult, 0)
	result := make([]vector.SearchResult, 0)

	d := idx.metric.Distance(query, idx.nodes[entryPoint].Data)
	candidates = append(candidates, vector.SearchResult{ID: entryPoint, Distance: d})
	result = append(result, vector.SearchResult{ID: entryPoint, Distance: d})
	visited[entryPoint] = true

	for len(candidates) > 0 {
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].Distance < candidates[j].Distance
		})
		curr := candidates[0]
		candidates = candidates[1:]

		if len(result) > 0 {
			sort.Slice(result, func(i, j int) bool {
				return result[i].Distance > result[j].Distance
			})
			if curr.Distance > result[0].Distance {
				break
			}
		}

		node := idx.nodes[curr.ID]
		if layer < len(node.Edges) {
			for _, neighborID := range node.Edges[layer] {
				if !visited[neighborID] {
					visited[neighborID] = true
					d := idx.metric.Distance(query, idx.nodes[neighborID].Data)

					if len(result) < ef || d < result[0].Distance {
						candidates = append(candidates, vector.SearchResult{ID: neighborID, Distance: d})
						result = append(result, vector.SearchResult{ID: neighborID, Distance: d})

						if len(result) > ef {
							sort.Slice(result, func(i, j int) bool {
								return result[i].Distance < result[j].Distance
							})
							result = result[:ef]
						}
					}
				}
			}
		}
	}
	return result
}

func (idx *Index) selectNeighbors(candidates []vector.SearchResult, M int) []int64 {
	if len(candidates) <= M {
		result := make([]int64, len(candidates))
		for i, c := range candidates {
			result[i] = c.ID
		}
		return result
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Distance < candidates[j].Distance
	})

	result := make([]int64, M)
	for i := 0; i < M; i++ {
		result[i] = candidates[i].ID
	}
	return result
}

func (idx *Index) pruneConnections(nodeID int64, layer, M int) {
	node := idx.nodes[nodeID]
	candList := make([]vector.SearchResult, len(node.Edges[layer]))
	for i, nid := range node.Edges[layer] {
		d := idx.metric.Distance(node.Data, idx.nodes[nid].Data)
		candList[i] = vector.SearchResult{ID: nid, Distance: d}
	}
	sort.Slice(candList, func(i, j int) bool {
		return candList[i].Distance < candList[j].Distance
	})
	node.Edges[layer] = make([]int64, M)
	for i := 0; i < M && i < len(candList); i++ {
		node.Edges[layer][i] = candList[i].ID
	}
}
