package main

import (
	"fmt"
	"time"

	"tinyllm/config"
	"tinyllm/gateway/server"
)

// apiVersionConfig shares the engine's [ext.api] migration-window format.
func apiVersionConfig(cfg config.Config) (server.APIVersionConfig, error) {
	out := server.APIVersionConfig{Default: "1"}
	raw, present := cfg.Ext["api"]
	if !present {
		return out, nil
	}
	api, ok := raw.(map[string]any)
	if !ok {
		return out, fmt.Errorf("ext.api must be a table")
	}
	for key, value := range api {
		switch key {
		case "default_version":
			v, ok := value.(string)
			if !ok || (v != "1" && v != "2") {
				return out, fmt.Errorf("ext.api.default_version must be the string 1 or 2")
			}
			out.Default = v
		case "deprecated_at", "sunset":
			v, ok := value.(int64)
			if !ok || v <= 0 {
				return out, fmt.Errorf("ext.api.%s must be Unix seconds > 0", key)
			}
			if key == "deprecated_at" {
				out.DeprecatedAt = time.Unix(v, 0)
			} else {
				out.Sunset = time.Unix(v, 0)
			}
		default:
			return out, fmt.Errorf("unknown key ext.api.%s", key)
		}
	}
	if out.DeprecatedAt.IsZero() != out.Sunset.IsZero() {
		return out, fmt.Errorf("ext.api.deprecated_at and ext.api.sunset open the window together")
	}
	if !out.Sunset.IsZero() && !out.Sunset.After(out.DeprecatedAt) {
		return out, fmt.Errorf("ext.api.sunset must come after ext.api.deprecated_at")
	}
	return out, nil
}
