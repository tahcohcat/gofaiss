package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
    "bufio"
	"strconv"
    "strings"

	"github.com/tahcohcat/gofaiss/pkg/index/hnsw"
	"github.com/tahcohcat/gofaiss/pkg/vector"
)

// HF API endpoint and token
const (
	HFToken = ""
	HFURL   = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
)

// loadCSVEmbeddings loads precomputed embeddings from CSV
func loadCSVEmbeddings(path string) ([][]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}

	embs := make([][]float32, len(records))
	for i, rec := range records {
		embs[i] = make([]float32, len(rec))
		for j, v := range rec {
			val, _ := strconv.ParseFloat(v, 32)
			embs[i][j] = float32(val)
		}
	}
	return embs, nil
}

// Review represents one parsed line
type Review struct {
	Text       string
	Restaurant string
	Cuisine    string
	City       string
}

// loadTexts loads texts from JSON
func loadTexts(path string) ([]Review, error) {
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	
	var reviews []Review
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, " | ")
		if len(parts) != 4 {
			fmt.Println("Skipping malformed line:", line)
			continue
		}

		r := Review{
			Text:       strings.TrimSpace(parts[0]),
			Restaurant: strings.TrimPrefix(strings.TrimSpace(parts[1]), "Restaurant: "),
			Cuisine:    strings.TrimPrefix(strings.TrimSpace(parts[2]), "Cuisine: "),
			City:       strings.TrimPrefix(strings.TrimSpace(parts[3]), "City: "),
		}

		reviews = append(reviews, r)
	}

	if err := scanner.Err(); err != nil {
		panic(err)
	}
	return reviews, nil
}

// getEmbedding calls Hugging Face API for a single query
func getEmbedding(query string) ([]float32, error) {
	input := map[string]interface{}{
		"inputs": query,
	}
	body, _ := json.Marshal(input)
	req, _ := http.NewRequest("POST", HFURL, bytes.NewBuffer(body))
	req.Header.Set("Authorization", "Bearer "+HFToken)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := ioutil.ReadAll(resp.Body)
	// Hugging Face returns {"embedding":[...]} or just [...] depending on model
    fmt.Printf("HF response: [%v]%s\n", resp, string(respBody)) // Debug print
	var result map[string][]float32
	if err := json.Unmarshal(respBody, &result); err != nil {
		// fallback: try parsing as []float32 directly
		var arr []float32
		if err2 := json.Unmarshal(respBody, &arr); err2 != nil {
			return nil, fmt.Errorf("failed to parse HF response: %v", err)
		}
		return arr, nil
	}
	return result["embedding"], nil
}

func main() {
	// Load precomputed embeddings
	embs, err := loadCSVEmbeddings("examples/kaggle_foodpanda_reviews/data/embeddings_1k.csv")
	if err != nil {
		panic(err)
	}

	// Load corresponding texts
	reviews, err := loadTexts("examples/kaggle_foodpanda_reviews/data/texts_1k.csv")
	if err != nil {
		panic(err)
	}

	// Build FAISS index
	dim := len(embs[0])
	index, err := hnsw.New(dim, "cosine", hnsw.DefaultConfig())
	if err != nil {
		panic(err)
	}

	// Convert [][]float32 to []vector.Vector
	vectors := make([]vector.Vector, len(embs))
	for i, emb := range embs {
		vectors[i] = vector.Vector{
			ID:   int64(i),
			Data: emb,
		}
	}

	err = index.Add(vectors)
	if err != nil {
		panic(err)
	}

	fmt.Println("Go-FAISS demo ready! Type a query:")

	// Read query from user
	var query string
	fmt.Print("> ")
	fmt.Scanln(&query)

	// Get embedding from Hugging Face
	qEmb, err := getEmbedding(query)
	if err != nil {
		panic(err)
	}

	// Search FAISS
	k := 5
	results, err := index.Search(qEmb, k)
	if err != nil {
		panic(err)
	}

	fmt.Println("\nTop matches:")
	for i, result := range results {
		if int(result.ID) < len(reviews) {
			fmt.Printf("%d. %s (distance %.4f)\n", i+1, reviews[result.ID], result.Distance)
		}
	}
}
