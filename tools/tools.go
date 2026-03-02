package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/rabidclock/localfreshha/haclient"
)

// ToolCall represents an LLM's request to invoke a tool.
type ToolCall struct {
	ID   string
	Name string
	Args map[string]any
}

// ToolResult is the output returned to the LLM after executing a tool.
type ToolResult struct {
	ID      string
	Name    string
	Content string
	IsError bool
}

// toolDef is a provider-agnostic tool definition.
type toolDef struct {
	Name        string
	Description string
	Properties  map[string]map[string]string
	Required    []string
}

var allTools = []toolDef{
	{
		Name:        "ha_get_entities",
		Description: "List all Home Assistant entities in a specific domain (e.g. light, switch, climate, sensor). Returns entity IDs, states, and friendly names.",
		Properties: map[string]map[string]string{
			"domain": {"type": "string", "description": "The entity domain to list (light, switch, climate, sensor, binary_sensor, input_boolean)"},
		},
		Required: []string{"domain"},
	},
	{
		Name:        "ha_get_state",
		Description: "Get the current state and attributes of a specific Home Assistant entity. Returns state value, brightness, temperature, and other attributes.",
		Properties: map[string]map[string]string{
			"entity_id": {"type": "string", "description": "The entity ID (e.g. light.living_room, climate.thermostat)"},
		},
		Required: []string{"entity_id"},
	},
	{
		Name:        "ha_turn_on",
		Description: "Turn on a light or switch in Home Assistant. Optionally set brightness (0-255) and color temperature (in mireds).",
		Properties: map[string]map[string]string{
			"entity_id":  {"type": "string", "description": "The entity ID to turn on (e.g. light.living_room)"},
			"brightness": {"type": "number", "description": "Brightness level 0-255 (optional, lights only)"},
			"color_temp": {"type": "number", "description": "Color temperature in mireds (optional, lights only)"},
		},
		Required: []string{"entity_id"},
	},
	{
		Name:        "ha_turn_off",
		Description: "Turn off a light or switch in Home Assistant.",
		Properties: map[string]map[string]string{
			"entity_id": {"type": "string", "description": "The entity ID to turn off (e.g. light.living_room)"},
		},
		Required: []string{"entity_id"},
	},
	{
		Name:        "ha_set_temperature",
		Description: "Set the target temperature on a thermostat. Optionally set the HVAC mode (heat, cool, auto, off).",
		Properties: map[string]map[string]string{
			"entity_id":   {"type": "string", "description": "The climate entity ID (e.g. climate.thermostat)"},
			"temperature": {"type": "number", "description": "Target temperature in the thermostat's configured unit"},
			"hvac_mode":   {"type": "string", "description": "HVAC mode: heat, cool, auto, or off (optional)"},
		},
		Required: []string{"entity_id", "temperature"},
	},
}

func buildProps(props map[string]map[string]string) map[string]any {
	out := make(map[string]any, len(props))
	for name, p := range props {
		out[name] = map[string]any{"type": p["type"], "description": p["description"]}
	}
	return out
}

// OllamaToolDefs returns tool definitions in Ollama's format.
func OllamaToolDefs() []any {
	out := make([]any, 0, len(allTools))
	for _, t := range allTools {
		params := map[string]any{
			"type":       "object",
			"properties": buildProps(t.Properties),
		}
		if len(t.Required) > 0 {
			params["required"] = t.Required
		}
		out = append(out, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        t.Name,
				"description": t.Description,
				"parameters":  params,
			},
		})
	}
	return out
}

// AnthropicToolDefs returns tool definitions in Anthropic's format.
func AnthropicToolDefs() []any {
	out := make([]any, 0, len(allTools))
	for _, t := range allTools {
		schema := map[string]any{
			"type":       "object",
			"properties": buildProps(t.Properties),
		}
		if len(t.Required) > 0 {
			schema["required"] = t.Required
		}
		out = append(out, map[string]any{
			"name":         t.Name,
			"description":  t.Description,
			"input_schema": schema,
		})
	}
	return out
}

// ToolNames returns the set of tool names provided by this package.
func ToolNames() map[string]bool {
	names := make(map[string]bool, len(allTools))
	for _, t := range allTools {
		names[t.Name] = true
	}
	return names
}

// Executor runs Home Assistant tool calls and returns results.
type Executor struct {
	client *haclient.Client
}

// NewExecutor creates a tool executor with the given HA client.
func NewExecutor(client *haclient.Client) *Executor {
	return &Executor{client: client}
}

// Execute runs a single tool call and returns the result.
// All outputs are sanitized and wrapped to prevent prompt injection.
func (e *Executor) Execute(ctx context.Context, call ToolCall) ToolResult {
	var result ToolResult
	switch call.Name {
	case "ha_get_entities":
		result = e.execGetEntities(ctx, call)
	case "ha_get_state":
		result = e.execGetState(ctx, call)
	case "ha_turn_on":
		result = e.execTurnOn(ctx, call)
	case "ha_turn_off":
		result = e.execTurnOff(ctx, call)
	case "ha_set_temperature":
		result = e.execSetTemperature(ctx, call)
	default:
		return ToolResult{
			ID:      call.ID,
			Name:    call.Name,
			Content: fmt.Sprintf("unknown tool: %s", call.Name),
			IsError: true,
		}
	}

	result.Content = sanitize(result.Content)
	if !result.IsError {
		result.Content = wrapToolOutput(result.Name, result.Content)
	}
	return result
}

func (e *Executor) execGetEntities(ctx context.Context, call ToolCall) ToolResult {
	domain, _ := call.Args["domain"].(string)
	if domain == "" {
		return ToolResult{ID: call.ID, Name: call.Name, Content: "missing required parameter: domain", IsError: true}
	}
	if !haclient.AllowedDomains[domain] {
		return ToolResult{ID: call.ID, Name: call.Name, Content: fmt.Sprintf("domain %q is not allowed (allowed: light, switch, climate, sensor, binary_sensor, input_boolean)", domain), IsError: true}
	}

	states, err := e.client.GetStates(ctx)
	if err != nil {
		return ToolResult{ID: call.ID, Name: call.Name, Content: fmt.Sprintf("error fetching states: %v", err), IsError: true}
	}

	var sb strings.Builder
	prefix := domain + "."
	count := 0
	for _, s := range states {
		if strings.HasPrefix(s.EntityID, prefix) {
			name, _ := s.Attributes["friendly_name"].(string)
			if name == "" {
				name = s.EntityID
			}
			fmt.Fprintf(&sb, "- %s: %s (%s)\n", s.EntityID, s.State, truncate(name, 100))
			count++
		}
	}

	if count == 0 {
		return ToolResult{ID: call.ID, Name: call.Name, Content: fmt.Sprintf("no entities found in domain %q", domain)}
	}

	return ToolResult{ID: call.ID, Name: call.Name, Content: fmt.Sprintf("Found %d %s entities:\n%s", count, domain, sb.String())}
}

func (e *Executor) execGetState(ctx context.Context, call ToolCall) ToolResult {
	entityID, _ := call.Args["entity_id"].(string)
	if entityID == "" {
		return ToolResult{ID: call.ID, Name: call.Name, Content: "missing required parameter: entity_id", IsError: true}
	}

	state, err := e.client.GetState(ctx, entityID)
	if err != nil {
		return ToolResult{ID: call.ID, Name: call.Name, Content: fmt.Sprintf("error: %v", err), IsError: true}
	}

	return ToolResult{ID: call.ID, Name: call.Name, Content: formatState(state)}
}

func (e *Executor) execTurnOn(ctx context.Context, call ToolCall) ToolResult {
	entityID, _ := call.Args["entity_id"].(string)
	if entityID == "" {
		return ToolResult{ID: call.ID, Name: call.Name, Content: "missing required parameter: entity_id", IsError: true}
	}
	if err := haclient.ValidateEntityID(entityID); err != nil {
		return ToolResult{ID: call.ID, Name: call.Name, Content: err.Error(), IsError: true}
	}

	domain := entityID[:strings.Index(entityID, ".")]
	if domain != "light" && domain != "switch" && domain != "input_boolean" {
		return ToolResult{ID: call.ID, Name: call.Name, Content: fmt.Sprintf("turn_on not supported for domain %q", domain), IsError: true}
	}

	data := map[string]any{"entity_id": entityID}
	if b, ok := toFloat64(call.Args["brightness"]); ok {
		data["brightness"] = b
	}
	if ct, ok := toFloat64(call.Args["color_temp"]); ok {
		data["color_temp"] = ct
	}

	if err := e.client.CallService(ctx, domain, "turn_on", data); err != nil {
		return ToolResult{ID: call.ID, Name: call.Name, Content: fmt.Sprintf("error: %v", err), IsError: true}
	}

	// Read back state for confirmation.
	state, err := e.client.GetState(ctx, entityID)
	if err != nil {
		return ToolResult{ID: call.ID, Name: call.Name, Content: fmt.Sprintf("turned on %s but could not read back state: %v", entityID, err)}
	}

	return ToolResult{ID: call.ID, Name: call.Name, Content: fmt.Sprintf("Turned on %s. Current state:\n%s", entityID, formatState(state))}
}

func (e *Executor) execTurnOff(ctx context.Context, call ToolCall) ToolResult {
	entityID, _ := call.Args["entity_id"].(string)
	if entityID == "" {
		return ToolResult{ID: call.ID, Name: call.Name, Content: "missing required parameter: entity_id", IsError: true}
	}
	if err := haclient.ValidateEntityID(entityID); err != nil {
		return ToolResult{ID: call.ID, Name: call.Name, Content: err.Error(), IsError: true}
	}

	domain := entityID[:strings.Index(entityID, ".")]
	if domain != "light" && domain != "switch" && domain != "input_boolean" {
		return ToolResult{ID: call.ID, Name: call.Name, Content: fmt.Sprintf("turn_off not supported for domain %q", domain), IsError: true}
	}

	data := map[string]any{"entity_id": entityID}
	if err := e.client.CallService(ctx, domain, "turn_off", data); err != nil {
		return ToolResult{ID: call.ID, Name: call.Name, Content: fmt.Sprintf("error: %v", err), IsError: true}
	}

	state, err := e.client.GetState(ctx, entityID)
	if err != nil {
		return ToolResult{ID: call.ID, Name: call.Name, Content: fmt.Sprintf("turned off %s but could not read back state: %v", entityID, err)}
	}

	return ToolResult{ID: call.ID, Name: call.Name, Content: fmt.Sprintf("Turned off %s. Current state:\n%s", entityID, formatState(state))}
}

func (e *Executor) execSetTemperature(ctx context.Context, call ToolCall) ToolResult {
	entityID, _ := call.Args["entity_id"].(string)
	if entityID == "" {
		return ToolResult{ID: call.ID, Name: call.Name, Content: "missing required parameter: entity_id", IsError: true}
	}
	if err := haclient.ValidateEntityID(entityID); err != nil {
		return ToolResult{ID: call.ID, Name: call.Name, Content: err.Error(), IsError: true}
	}

	domain := entityID[:strings.Index(entityID, ".")]
	if domain != "climate" {
		return ToolResult{ID: call.ID, Name: call.Name, Content: fmt.Sprintf("set_temperature only works for climate entities, got %q", domain), IsError: true}
	}

	temp, ok := toFloat64(call.Args["temperature"])
	if !ok {
		return ToolResult{ID: call.ID, Name: call.Name, Content: "missing required parameter: temperature", IsError: true}
	}

	data := map[string]any{
		"entity_id":   entityID,
		"temperature": temp,
	}
	if mode, ok := call.Args["hvac_mode"].(string); ok && mode != "" {
		data["hvac_mode"] = mode
	}

	if err := e.client.CallService(ctx, "climate", "set_temperature", data); err != nil {
		return ToolResult{ID: call.ID, Name: call.Name, Content: fmt.Sprintf("error: %v", err), IsError: true}
	}

	state, err := e.client.GetState(ctx, entityID)
	if err != nil {
		return ToolResult{ID: call.ID, Name: call.Name, Content: fmt.Sprintf("set temperature to %.1f but could not read back state: %v", temp, err)}
	}

	return ToolResult{ID: call.ID, Name: call.Name, Content: fmt.Sprintf("Set temperature to %.1f. Current state:\n%s", temp, formatState(state))}
}

// formatState creates a human-readable string from an entity state.
func formatState(s *haclient.EntityState) string {
	var sb strings.Builder
	name, _ := s.Attributes["friendly_name"].(string)
	if name != "" {
		fmt.Fprintf(&sb, "Name: %s\n", name)
	}
	fmt.Fprintf(&sb, "Entity: %s\nState: %s\n", s.EntityID, s.State)

	// Include useful attributes.
	for _, key := range []string{
		"brightness", "color_temp", "color_mode",
		"temperature", "current_temperature", "hvac_mode", "hvac_action",
		"unit_of_measurement", "device_class",
	} {
		if v, ok := s.Attributes[key]; ok {
			fmt.Fprintf(&sb, "%s: %v\n", key, v)
		}
	}

	return sb.String()
}

// toFloat64 extracts a float64 from an any value (handles both float64 and json.Number).
func toFloat64(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	default:
		return 0, false
	}
}
