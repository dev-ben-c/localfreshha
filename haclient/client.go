package haclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"
)

var entityIDPattern = regexp.MustCompile(`^[a-z_]+\.[a-z0-9_]+$`)

// AllowedDomains is the set of HA entity domains that tools are permitted to access.
var AllowedDomains = map[string]bool{
	"light":          true,
	"switch":         true,
	"climate":        true,
	"sensor":         true,
	"binary_sensor":  true,
	"input_boolean":  true,
}

// EntityState represents a Home Assistant entity state.
type EntityState struct {
	EntityID    string         `json:"entity_id"`
	State       string         `json:"state"`
	Attributes  map[string]any `json:"attributes"`
	LastChanged string         `json:"last_changed"`
	LastUpdated string         `json:"last_updated"`
}

// Client communicates with the Home Assistant REST API.
type Client struct {
	baseURL    string
	token      string
	httpClient *http.Client
}

// NewClient creates a Client from HA_URL and HA_TOKEN environment variables.
// Returns an error if HA_TOKEN is not set.
func NewClient() (*Client, error) {
	token := os.Getenv("HA_TOKEN")
	if token == "" {
		return nil, fmt.Errorf("HA_TOKEN environment variable is not set")
	}

	baseURL := os.Getenv("HA_URL")
	if baseURL == "" {
		baseURL = "http://192.168.0.184:8123"
	}

	return &Client{
		baseURL: baseURL,
		token:   token,
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}, nil
}

// NewClientWithConfig creates a Client with explicit configuration.
// Useful for testing.
func NewClientWithConfig(baseURL, token string, httpClient *http.Client) *Client {
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 10 * time.Second}
	}
	return &Client{
		baseURL:    baseURL,
		token:      token,
		httpClient: httpClient,
	}
}

// ValidateEntityID checks that an entity ID is well-formed and in an allowed domain.
func ValidateEntityID(entityID string) error {
	if !entityIDPattern.MatchString(entityID) {
		return fmt.Errorf("invalid entity ID format: %q", entityID)
	}
	dot := strings.Index(entityID, ".")
	if dot < 0 {
		return fmt.Errorf("invalid entity ID format: %q", entityID)
	}
	domain := entityID[:dot]
	if !AllowedDomains[domain] {
		return fmt.Errorf("domain %q is not allowed (allowed: light, switch, climate, sensor, binary_sensor, input_boolean)", domain)
	}
	return nil
}

// GetStates returns all entity states from Home Assistant.
func (c *Client) GetStates(ctx context.Context) ([]EntityState, error) {
	body, err := c.doGet(ctx, "/api/states")
	if err != nil {
		return nil, err
	}
	var states []EntityState
	if err := json.Unmarshal(body, &states); err != nil {
		return nil, fmt.Errorf("parsing states: %w", err)
	}
	return states, nil
}

// GetState returns the state of a specific entity.
func (c *Client) GetState(ctx context.Context, entityID string) (*EntityState, error) {
	if err := ValidateEntityID(entityID); err != nil {
		return nil, err
	}
	body, err := c.doGet(ctx, "/api/states/"+entityID)
	if err != nil {
		return nil, err
	}
	var state EntityState
	if err := json.Unmarshal(body, &state); err != nil {
		return nil, fmt.Errorf("parsing state: %w", err)
	}
	return &state, nil
}

// CallService calls a Home Assistant service.
func (c *Client) CallService(ctx context.Context, domain, service string, data map[string]any) error {
	payload, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("marshalling service data: %w", err)
	}

	url := fmt.Sprintf("%s/api/services/%s/%s", c.baseURL, domain, service)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("calling service %s/%s: %w", domain, service, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("service %s/%s returned %d: %s", domain, service, resp.StatusCode, string(body))
	}
	return nil
}

func (c *Client) doGet(ctx context.Context, path string) ([]byte, error) {
	url := c.baseURL + path
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.token)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("GET %s: %w", path, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("GET %s returned %d: %s", path, resp.StatusCode, string(body))
	}
	return body, nil
}
