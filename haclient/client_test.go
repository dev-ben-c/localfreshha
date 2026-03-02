package haclient

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func testServer(t *testing.T, handler http.HandlerFunc) (*Client, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	client := NewClientWithConfig(srv.URL, "test-token", srv.Client())
	return client, srv
}

func TestGetStates(t *testing.T) {
	states := []EntityState{
		{EntityID: "light.living_room", State: "on", Attributes: map[string]any{"brightness": float64(255)}},
		{EntityID: "sensor.temperature", State: "72", Attributes: map[string]any{"unit_of_measurement": "°F"}},
	}

	client, srv := testServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/states" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-token" {
			t.Error("missing or wrong auth header")
		}
		json.NewEncoder(w).Encode(states)
	})
	defer srv.Close()

	got, err := client.GetStates(context.Background())
	if err != nil {
		t.Fatalf("GetStates: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 states, got %d", len(got))
	}
	if got[0].EntityID != "light.living_room" {
		t.Errorf("expected light.living_room, got %s", got[0].EntityID)
	}
}

func TestGetState(t *testing.T) {
	state := EntityState{
		EntityID:   "light.living_room",
		State:      "on",
		Attributes: map[string]any{"brightness": float64(200)},
	}

	client, srv := testServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/states/light.living_room" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		json.NewEncoder(w).Encode(state)
	})
	defer srv.Close()

	got, err := client.GetState(context.Background(), "light.living_room")
	if err != nil {
		t.Fatalf("GetState: %v", err)
	}
	if got.State != "on" {
		t.Errorf("expected on, got %s", got.State)
	}
}

func TestGetStateInvalidEntityID(t *testing.T) {
	client := NewClientWithConfig("http://localhost", "token", nil)

	_, err := client.GetState(context.Background(), "invalid")
	if err == nil {
		t.Fatal("expected error for invalid entity ID")
	}
}

func TestGetStateForbiddenDomain(t *testing.T) {
	client := NewClientWithConfig("http://localhost", "token", nil)

	_, err := client.GetState(context.Background(), "lock.front_door")
	if err == nil {
		t.Fatal("expected error for disallowed domain")
	}
}

func TestCallService(t *testing.T) {
	client, srv := testServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/services/light/turn_on" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["entity_id"] != "light.living_room" {
			t.Errorf("expected light.living_room in body, got %v", body["entity_id"])
		}
		w.WriteHeader(http.StatusOK)
	})
	defer srv.Close()

	err := client.CallService(context.Background(), "light", "turn_on", map[string]any{
		"entity_id": "light.living_room",
	})
	if err != nil {
		t.Fatalf("CallService: %v", err)
	}
}

func TestCallServiceError(t *testing.T) {
	client, srv := testServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"message":"entity not found"}`))
	})
	defer srv.Close()

	err := client.CallService(context.Background(), "light", "turn_on", map[string]any{
		"entity_id": "light.nonexistent",
	})
	if err == nil {
		t.Fatal("expected error for 400 response")
	}
}

func TestValidateEntityID(t *testing.T) {
	tests := []struct {
		id      string
		wantErr bool
	}{
		{"light.living_room", false},
		{"switch.bedroom", false},
		{"climate.thermostat", false},
		{"sensor.temperature", false},
		{"binary_sensor.motion", false},
		{"input_boolean.vacation", false},
		{"lock.front_door", true},
		{"automation.test", true},
		{"invalid", true},
		{"LIGHT.ROOM", true},
		{"light.room; DROP TABLE", true},
		{"", true},
	}

	for _, tt := range tests {
		err := ValidateEntityID(tt.id)
		if (err != nil) != tt.wantErr {
			t.Errorf("ValidateEntityID(%q) error=%v, wantErr=%v", tt.id, err, tt.wantErr)
		}
	}
}
