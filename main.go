package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
)

const (
	defaultConfigPath = "config.json"
	defaultMaxTokens  = 1024
)

type Config struct {
	APIKey  string `json:"api_key"`
	BaseURL string `json:"base_url"`
	Model   string `json:"model"`
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type llmClient interface {
	runInference(ctx context.Context, conversation []ChatMessage) (ChatMessage, error)
}

type Agent struct {
	client         llmClient
	getUserMessage func() (string, bool)
	tools          []ToolDefinition
}

type openAIClient struct {
	httpClient  *http.Client
	endpointURL string
	apiKey      string
	model       string
	tools       []openAIChatTool
}

type openAIChatRequest struct {
	Model     string           `json:"model"`
	Messages  []ChatMessage    `json:"messages"`
	MaxTokens int              `json:"max_tokens"`
	Tools     []openAIChatTool `json:"tools,omitempty"`
}

type openAIChatResponse struct {
	Choices []struct {
		Message ChatMessage `json:"message"`
	} `json:"choices"`
}

type openAIChatTool struct {
	Type     string                 `json:"type"`
	Function openAIChatToolFunction `json:"function"`
}

type openAIChatToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

func main() {
	cfg, err := loadConfig(defaultConfigPath)
	if err != nil {
		log.Fatalf("load config: %v", err)
	}

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	tools := []ToolDefinition{}
	agent := NewAgent(newOpenAIClient(cfg, nil, tools), getUserMessage, tools)
	if err := agent.Run(context.Background()); err != nil {
		log.Fatalf("run agent: %v", err)
	}
}

func loadConfig(path string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("read %s: %w", path, err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse %s: %w", path, err)
	}

	cfg.APIKey = strings.TrimSpace(cfg.APIKey)
	cfg.BaseURL = strings.TrimSpace(cfg.BaseURL)
	cfg.Model = strings.TrimSpace(cfg.Model)

	switch {
	case cfg.APIKey == "":
		return Config{}, fmt.Errorf("config %s missing api_key", path)
	case cfg.BaseURL == "":
		return Config{}, fmt.Errorf("config %s missing base_url", path)
	case cfg.Model == "":
		return Config{}, fmt.Errorf("config %s missing model", path)
	}

	return cfg, nil
}

func NewAgent(client llmClient, getUserMessage func() (string, bool), tools []ToolDefinition) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
		tools:          tools,
	}
}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []ChatMessage{}
	conversation = append(conversation, ChatMessage{
		Role:    "system",
		Content: "You are a helpful assistant. Answer the user's questions to the best of your ability.",
	})
	log.Printf("Starting agent. Type your messages below:")

	for {
		fmt.Print("You: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			log.Printf("No more user input. Exiting.")
			return nil
		}

		conversation = append(conversation, ChatMessage{
			Role:    "user",
			Content: userInput,
		})

		reply, err := a.client.runInference(ctx, conversation)
		if err != nil {
			return err
		}

		conversation = append(conversation, reply)
		log.Printf("Agent: %s", reply.Content)
	}
}

func newOpenAIClient(cfg Config, httpClient *http.Client, tools []ToolDefinition) *openAIClient {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	return &openAIClient{
		httpClient:  httpClient,
		endpointURL: buildChatCompletionsURL(cfg.BaseURL),
		apiKey:      cfg.APIKey,
		model:       cfg.Model,
		tools:       buildOpenAIChatTools(tools),
	}
}

func (c *openAIClient) runInference(ctx context.Context, conversation []ChatMessage) (ChatMessage, error) {
	requestBody, err := json.Marshal(openAIChatRequest{
		Model:     c.model,
		Messages:  conversation,
		MaxTokens: defaultMaxTokens,
		Tools:     c.tools,
	})
	if err != nil {
		return ChatMessage{}, fmt.Errorf("marshal chat request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpointURL, bytes.NewReader(requestBody))
	if err != nil {
		return ChatMessage{}, fmt.Errorf("build chat request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return ChatMessage{}, fmt.Errorf("send chat request: %w", err)
	}
	defer resp.Body.Close()

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return ChatMessage{}, fmt.Errorf("read chat response: %w", err)
	}

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return ChatMessage{}, fmt.Errorf("chat completion request failed with status %d: %s", resp.StatusCode, strings.TrimSpace(string(responseBody)))
	}

	var payload openAIChatResponse
	if err := json.Unmarshal(responseBody, &payload); err != nil {
		return ChatMessage{}, fmt.Errorf("decode chat response: %w", err)
	}
	if len(payload.Choices) == 0 {
		return ChatMessage{}, fmt.Errorf("chat response did not include any choices")
	}

	reply := payload.Choices[0].Message
	if strings.TrimSpace(reply.Role) == "" {
		reply.Role = "assistant"
	}
	return reply, nil
}

func buildChatCompletionsURL(baseURL string) string {
	trimmed := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	switch {
	case strings.HasSuffix(trimmed, "/chat/completions"):
		return trimmed
	case strings.HasSuffix(trimmed, "/v1"):
		return trimmed + "/chat/completions"
	default:
		return trimmed + "/v1/chat/completions"
	}
}

func buildOpenAIChatTools(definitions []ToolDefinition) []openAIChatTool {
	if len(definitions) == 0 {
		return nil
	}

	tools := make([]openAIChatTool, 0, len(definitions))
	for _, definition := range definitions {
		tools = append(tools, openAIChatTool{
			Type: "function",
			Function: openAIChatToolFunction{
				Name:        definition.Name,
				Description: definition.Description,
				Parameters:  definition.Parameters,
			},
		})
	}

	return tools
}

type ToolDefinition struct {
	Name        string                                   `json:"name"`
	Description string                                   `json:"description"`
	Parameters  json.RawMessage                          `json:"parameters,omitempty"`
	Function    func(in json.RawMessage) (string, error) `json:"-"`
}
