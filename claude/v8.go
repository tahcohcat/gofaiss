package faiss

import (
	"bytes"
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
	"time"
)

// Vector represents a single vector with an ID
type Vector struct {
	ID   int64
	Data []float32
	Norm float32
}

// SearchResult represents a search result with distance
type SearchResult struct {
	ID       int64
	Distance float32
}

// Index interface defines the basic operations for vector indexing
type Index interface {
	Add(vectors []Vector) error
	Search(query []float32, k int) ([]SearchResult, error)
	Remove(id int64) error
	Size() int
}

// FlatIndex implements a simple flat (brute-force) index
type FlatIndex struct {
	dim     int
	metric  string
	vectors []Vector
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
		vectors: make([]Vector, 0),
	}, nil
}

func (idx *FlatIndex) Add(vectors []Vector) error {
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

func (idx *FlatIndex) Search(query []float32, k int) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(query) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch")
	}
	if k <= 0 || k > len(idx.vectors) {
		k = len(idx.vectors)
	}

	queryNorm := float32(0)
	if idx.metric == "cosine" {
		queryNorm = computeNorm(query)
		if queryNorm == 0 {
			return nil, fmt.Errorf("zero query vector not allowed")
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

func (idx *FlatIndex) Remove(id int64) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for i, v := range idx.vectors {
		if v.ID == id {
			idx.vectors = append(idx.vectors[:i], idx.vectors[i+1:]...)
			return nil
		}
	}
	return fmt.Errorf("vector with id %d not found", id)
}

func (idx *FlatIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.vectors)
}

func (idx *FlatIndex) BatchSearch(queries [][]float32, k int) ([][]SearchResult, error) {
	results := make([][]SearchResult, len(queries))
	for i, query := range queries {
		res, err := idx.Search(query, k)
		if err != nil {
			return nil, err
		}
		results[i] = res
	}
	return results, nil
}

// Distance functions
func l2Distance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

func cosineDistance(a, b []float32, normA, normB float32) float32 {
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}
	similarity := dot / (normA * normB)
	if similarity > 1 {
		similarity = 1
	} else if similarity < -1 {
		similarity = -1
	}
	return 1 - similarity
}

func computeNorm(v []float32) float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	return float32(math.Sqrt(float64(sum)))
}

// ProductQuantizer implements product quantization
type ProductQuantizer struct {
	dim       int
	M         int
	nbits     int
	dsub      int
	ksub      int
	codebooks [][]Vector
	trained   bool
	mu        sync.RWMutex
}

func NewProductQuantizer(dim int, M int, nbits int) (*ProductQuantizer, error) {
	if dim%M != 0 {
		return nil, fmt.Errorf("dimension %d must be divisible by M %d", dim, M)
	}
	if nbits <= 0 || nbits > 16 {
		return nil, fmt.Errorf("nbits must be between 1 and 16")
	}

	dsub := dim / M
	ksub := 1 << nbits

	return &ProductQuantizer{
		dim:       dim,
		M:         M,
		nbits:     nbits,
		dsub:      dsub,
		ksub:      ksub,
		codebooks: make([][]Vector, M),
	}, nil
}

func (pq *ProductQuantizer) Train(vectors []Vector) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	if len(vectors) < pq.ksub {
		return fmt.Errorf("need at least %d training vectors", pq.ksub)
	}

	for m := 0; m < pq.M; m++ {
		start := m * pq.dsub
		end := start + pq.dsub

		subvectors := make([]Vector, len(vectors))
		for i, v := range vectors {
			subvectors[i] = Vector{
				ID:   v.ID,
				Data: v.Data[start:end],
			}
		}

		pq.codebooks[m] = kMeans(subvectors, pq.ksub, "l2", 20)
	}

	pq.trained = true
	return nil
}

func (pq *ProductQuantizer) Encode(v []float32) ([]uint16, error) {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	if !pq.trained {
		return nil, fmt.Errorf("quantizer must be trained")
	}
	if len(v) != pq.dim {
		return nil, fmt.Errorf("vector dimension mismatch")
	}

	codes := make([]uint16, pq.M)
	for m := 0; m < pq.M; m++ {
		start := m * pq.dsub
		end := start + pq.dsub
		subvec := v[start:end]

		minDist := float32(math.Inf(1))
		minIdx := 0
		for k, centroid := range pq.codebooks[m] {
			dist := l2Distance(subvec, centroid.Data)
			if dist < minDist {
				minDist = dist
				minIdx = k
			}
		}
		codes[m] = uint16(minIdx)
	}
	return codes, nil
}

func (pq *ProductQuantizer) ComputeDistanceTable(query []float32) ([][]float32, error) {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	if !pq.trained {
		return nil, fmt.Errorf("quantizer must be trained")
	}

	table := make([][]float32, pq.M)
	for m := 0; m < pq.M; m++ {
		start := m * pq.dsub
		end := start + pq.dsub
		subquery := query[start:end]

		table[m] = make([]float32, pq.ksub)
		for k, centroid := range pq.codebooks[m] {
			table[m][k] = l2Distance(subquery, centroid.Data)
		}
	}
	return table, nil
}

func ComputePQDistance(codes []uint16, table [][]float32) float32 {
	var sum float32
	for m, code := range codes {
		dist := table[m][code]
		sum += dist * dist
	}
	return float32(math.Sqrt(float64(sum)))
}

// PQVector stores quantized vector
type PQVector struct {
	ID    int64
	Codes []uint16
}

// IVFPQIndex combines IVF with PQ
type IVFPQIndex struct {
	dim       int
	metric    string
	nlist     int
	pq        *ProductQuantizer
	centroids []Vector
	lists     [][]PQVector
	mu        sync.RWMutex
	trained   bool
}

func NewIVFPQIndex(dim int, metric string, nlist int, M int, nbits int) (*IVFPQIndex, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	if metric != "l2" && metric != "cosine" {
		return nil, fmt.Errorf("metric must be 'l2' or 'cosine'")
	}
	if nlist <= 0 {
		return nil, fmt.Errorf("nlist must be positive")
	}

	pq, err := NewProductQuantizer(dim, M, nbits)
	if err != nil {
		return nil, err
	}

	return &IVFPQIndex{
		dim:    dim,
		metric: metric,
		nlist:  nlist,
		pq:     pq,
		lists:  make([][]PQVector, nlist),
	}, nil
}

func (idx *IVFPQIndex) Train(vectors []Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(vectors) < idx.nlist {
		return fmt.Errorf("need at least %d vectors to train", idx.nlist)
	}

	idx.centroids = kMeans(vectors, idx.nlist, idx.metric, 10)

	residuals := make([]Vector, len(vectors))
	for i, v := range vectors {
		nearestIdx := idx.findNearestCentroidUnsafe(v.Data)
		centroid := idx.centroids[nearestIdx].Data

		residual := make([]float32, idx.dim)
		for d := 0; d < idx.dim; d++ {
			residual[d] = v.Data[d] - centroid[d]
		}
		residuals[i] = Vector{ID: v.ID, Data: residual}
	}

	err := idx.pq.Train(residuals)
	if err != nil {
		return err
	}

	idx.trained = true
	return nil
}

func (idx *IVFPQIndex) Add(vectors []Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if !idx.trained {
		return fmt.Errorf("index must be trained")
	}

	for _, v := range vectors {
		if len(v.Data) != idx.dim {
			return fmt.Errorf("vector dimension mismatch")
		}

		nearestList := idx.findNearestCentroidUnsafe(v.Data)
		centroid := idx.centroids[nearestList].Data

		residual := make([]float32, idx.dim)
		for d := 0; d < idx.dim; d++ {
			residual[d] = v.Data[d] - centroid[d]
		}

		codes, err := idx.pq.Encode(residual)
		if err != nil {
			return err
		}

		idx.lists[nearestList] = append(idx.lists[nearestList], PQVector{
			ID:    v.ID,
			Codes: codes,
		})
	}
	return nil
}

func (idx *IVFPQIndex) Search(query []float32, k int, nprobe int) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if !idx.trained {
		return nil, fmt.Errorf("index must be trained")
	}
	if len(query) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch")
	}
	if nprobe <= 0 || nprobe > idx.nlist {
		nprobe = idx.nlist
	}

	centroidDists := make([]SearchResult, len(idx.centroids))
	queryNorm := float32(0)
	if idx.metric == "cosine" {
		queryNorm = computeNorm(query)
	}

	for i, c := range idx.centroids {
		var dist float32
		if idx.metric == "l2" {
			dist = l2Distance(query, c.Data)
		} else {
			dist = cosineDistance(query, c.Data, queryNorm, c.Norm)
		}
		centroidDists[i] = SearchResult{ID: int64(i), Distance: dist}
	}

	sort.Slice(centroidDists, func(i, j int) bool {
		return centroidDists[i].Distance < centroidDists[j].Distance
	})

	var results []SearchResult
	for i := 0; i < nprobe; i++ {
		listIdx := int(centroidDists[i].ID)
		if len(idx.lists[listIdx]) == 0 {
			continue
		}

		centroid := idx.centroids[listIdx].Data
		residualQuery := make([]float32, idx.dim)
		for d := 0; d < idx.dim; d++ {
			residualQuery[d] = query[d] - centroid[d]
		}

		table, err := idx.pq.ComputeDistanceTable(residualQuery)
		if err != nil {
			return nil, err
		}

		for _, v := range idx.lists[listIdx] {
			dist := ComputePQDistance(v.Codes, table)
			results = append(results, SearchResult{ID: v.ID, Distance: dist})
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	if k < len(results) {
		return results[:k], nil
	}
	return results, nil
}

func (idx *IVFPQIndex) findNearestCentroidUnsafe(v []float32) int {
	minDist := float32(math.Inf(1))
	minIdx := 0

	for i, c := range idx.centroids {
		var dist float32
		if idx.metric == "l2" {
			dist = l2Distance(v, c.Data)
		} else {
			vNorm := computeNorm(v)
			dist = cosineDistance(v, c.Data, vNorm, c.Norm)
		}
		if dist < minDist {
			minDist = dist
			minIdx = i
		}
	}
	return minIdx
}

func (idx *IVFPQIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	total := 0
	for _, list := range idx.lists {
		total += len(list)
	}
	return total
}

func (idx *IVFPQIndex) Remove(id int64) error {
	return fmt.Errorf("remove not implemented for IVF+PQ")
}

func (idx *IVFPQIndex) BatchSearch(queries [][]float32, k int, nprobe int) ([][]SearchResult, error) {
	results := make([][]SearchResult, len(queries))
	for i, query := range queries {
		res, err := idx.Search(query, k, nprobe)
		if err != nil {
			return nil, err
		}
		results[i] = res
	}
	return results, nil
}

// HNSWIndex implements HNSW
type HNSWIndex struct {
	dim            int
	metric         string
	M              int
	efConstruction int
	efSearch       int
	maxLevel       int
	entryPoint     int64
	vectors        map[int64]*HNSWNode
	levelMult      float64
	mu             sync.RWMutex
	nextID         int64
}

type HNSWNode struct {
	ID    int64
	Data  []float32
	Level int
	Edges [][]int64
}

func NewHNSWIndex(dim int, metric string, M int, efConstruction int) (*HNSWIndex, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	if metric != "l2" && metric != "cosine" {
		return nil, fmt.Errorf("metric must be 'l2' or 'cosine'")
	}
	if M <= 0 {
		return nil, fmt.Errorf("M must be positive")
	}

	return &HNSWIndex{
		dim:            dim,
		metric:         metric,
		M:              M,
		efConstruction: efConstruction,
		efSearch:       efConstruction,
		maxLevel:       0,
		entryPoint:     -1,
		vectors:        make(map[int64]*HNSWNode),
		levelMult:      1.0 / math.Log(float64(M)),
		nextID:         0,
	}, nil
}

func (idx *HNSWIndex) SetEfSearch(ef int) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.efSearch = ef
}

func (idx *HNSWIndex) Add(vectors []Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for _, v := range vectors {
		if len(v.Data) != idx.dim {
			return fmt.Errorf("vector dimension mismatch")
		}

		id := v.ID
		if id == 0 {
			id = idx.nextID
			idx.nextID++
		}

		level := idx.randomLevel()
		if level > idx.maxLevel {
			idx.maxLevel = level
		}

		node := &HNSWNode{
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
			idx.vectors[id] = node
			continue
		}

		idx.insertNode(node)
		idx.vectors[id] = node
	}
	return nil
}

func (idx *HNSWIndex) randomLevel() int {
	r := -math.Log(1.0 - float64(rand.Intn(1000000))/1000000.0)
	level := int(r * idx.levelMult)
	if level > 16 {
		level = 16
	}
	return level
}

func (idx *HNSWIndex) insertNode(node *HNSWNode) {
	curr := idx.entryPoint
	currDist := idx.distance(node.Data, idx.vectors[curr].Data)

	for lc := idx.maxLevel; lc > node.Level; lc-- {
		changed := true
		for changed {
			changed = false
			for _, neighborID := range idx.vectors[curr].Edges[lc] {
				d := idx.distance(node.Data, idx.vectors[neighborID].Data)
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
			idx.vectors[neighborID].Edges[lc] = append(idx.vectors[neighborID].Edges[lc], node.ID)

			if len(idx.vectors[neighborID].Edges[lc]) > M {
				neighborNode := idx.vectors[neighborID]
				candList := make([]SearchResult, len(neighborNode.Edges[lc]))
				for i, nid := range neighborNode.Edges[lc] {
					d := idx.distance(neighborNode.Data, idx.vectors[nid].Data)
					candList[i] = SearchResult{ID: nid, Distance: d}
				}
				sort.Slice(candList, func(i, j int) bool {
					return candList[i].Distance < candList[j].Distance
				})
				neighborNode.Edges[lc] = make([]int64, M)
				for i := 0; i < M; i++ {
					neighborNode.Edges[lc][i] = candList[i].ID
				}
			}
		}

		if len(candidates) > 0 {
			curr = candidates[0].ID
		}
	}
}

func (idx *HNSWIndex) searchLayer(query []float32, entryPoint int64, ef int, layer int) []SearchResult {
	visited := make(map[int64]bool)
	candidates := make([]SearchResult, 0)
	result := make([]SearchResult, 0)

	d := idx.distance(query, idx.vectors[entryPoint].Data)
	candidates = append(candidates, SearchResult{ID: entryPoint, Distance: d})
	result = append(result, SearchResult{ID: entryPoint, Distance: d})
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

		node := idx.vectors[curr.ID]
		if layer < len(node.Edges) {
			for _, neighborID := range node.Edges[layer] {
				if !visited[neighborID] {
					visited[neighborID] = true
					d := idx.distance(query, idx.vectors[neighborID].Data)

					if len(result) < ef || d < result[0].Distance {
						candidates = append(candidates, SearchResult{ID: neighborID, Distance: d})
						result = append(result, SearchResult{ID: neighborID, Distance: d})

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

func (idx *HNSWIndex) selectNeighbors(candidates []SearchResult, M int) []int64 {
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

func (idx *HNSWIndex) Search(query []float32, k int) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(query) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch")
	}
	if idx.entryPoint == -1 {
		return []SearchResult{}, nil
	}

	curr := idx.entryPoint
	currDist := idx.distance(query, idx.vectors[curr].Data)

	for lc := idx.maxLevel; lc > 0; lc-- {
		changed := true
		for changed {
			changed = false
			node := idx.vectors[curr]
			if lc < len(node.Edges) {
				for _, neighborID := range node.Edges[lc] {
					d := idx.distance(query, idx.vectors[neighborID].Data)
					if d < currDist {
						currDist = d
						curr = neighborID
						changed = true
					}
				}
			}
		}
	}

	candidates := idx.searchLayer(query, curr, idx.efSearch, 0)

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Distance < candidates[j].Distance
	})

	if k < len(candidates) {
		return candidates[:k], nil
	}
	return candidates, nil
}

func (idx *HNSWIndex) BatchSearch(queries [][]float32, k int) ([][]SearchResult, error) {
	results := make([][]SearchResult, len(queries))
	for i, query := range queries {
		res, err := idx.Search(query, k)
		if err != nil {
			return nil, err
		}
		results[i] = res
	}
	return results, nil
}

func (idx *HNSWIndex) distance(a, b []float32) float32 {
	if idx.metric == "l2" {
		return l2Distance(a, b)
	}
	normA := computeNorm(a)
	normB := computeNorm(b)
	return cosineDistance(a, b, normA, normB)
}

func (idx *HNSWIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.vectors)
}

func (idx *HNSWIndex) Remove(id int64) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	node, exists := idx.vectors[id]
	if !exists {
		return fmt.Errorf("node %d not found", id)
	}

	for _, otherNode := range idx.vectors {
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

	if idx.entryPoint == id {
		if len(idx.vectors) > 1 {
			for _, n := range idx.vectors {
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

	delete(idx.vectors, id)
	return nil
}

// k-means clustering
func kMeans(vectors []Vector, k int, metric string, maxIter int) []Vector {
	centroids := make([]Vector, k)
	step := len(vectors) / k
	for i := 0; i < k; i++ {
		idx := i * step
		if idx >= len(vectors) {
			idx = len(vectors) - 1
		}
		centroids[i] = Vector{
			ID:   int64(i),
			Data: make([]float32, len(vectors[idx].Data)),
		}
		copy(centroids[i].Data, vectors[idx].Data)
	}

	for iter := 0; iter < maxIter; iter++ {
		assignments := make([]int, len(vectors))
		for i, v := range vectors {
			minDist := float32(math.Inf(1))
			minIdx := 0
			for j, c := range centroids {
				dist := l2Distance(v.Data, c.Data)
				if dist < minDist {
					minDist = dist
					minIdx = j
				}
			}
			assignments[i] = minIdx
		}

		for i := range centroids {
			sum := make([]float32, len(centroids[i].Data))
			count := 0
			for j, a := range assignments {
				if a == i {
					for d := range sum {
						sum[d] += vectors[j].Data[d]
					}
					count++
				}
			}
			if count > 0 {
				for d := range sum {
					centroids[i].Data[d] = sum[d] / float32(count)
				}
			}
		}
	}

	if metric == "cosine" {
		for i := range centroids {
			centroids[i].Norm = computeNorm(centroids[i].Data)
		}
	}

	return centroids
}

// Serialization
func (idx *FlatIndex) SaveToFile(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	gzw := gzip.NewWriter(f)
	defer gzw.Close()

	idx.mu.R