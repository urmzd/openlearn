// Package config loads runtime.toml for every Go service (gw.01): defaults,
// then the file, then TL_<SECTION>__<KEY> environment overrides.
//
// Contract: course/contracts/config/runtime.schema.json: unknown
// keys are errors except under [ext]; every scalar key can be overridden by
// TL_<SECTION>__<KEY> (both upper-cased, a double underscore between them),
// parsed as the key's type; arrays and tables are not overridable.
// Precedence: environment, then file, then defaults.
// Chapter: content/courses/ai-platform-engineering/12-gateway/01-server-skeleton.md.
package config

import (
	"fmt"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"github.com/BurntSushi/toml"
)

// Config is the whole runtime.toml. Each service reads its own section plus
// [paths] and [otel].
type Config struct {
	Paths   Paths          `toml:"paths"`
	OTel    OTel           `toml:"otel"`
	Gateway Gateway        `toml:"gateway"`
	Engine  Engine         `toml:"engine"`
	Durable Durable        `toml:"durable"`
	Worker  Worker         `toml:"worker"`
	Agent   Agent          `toml:"agent"`
	Ext     map[string]any `toml:"ext"`
}

type Paths struct {
	Artifacts string `toml:"artifacts"`
}

type OTel struct {
	Endpoint         string  `toml:"endpoint"`
	ServiceNamespace string  `toml:"service_namespace"`
	TraceSampleRatio float64 `toml:"trace_sample_ratio"`
}

type Gateway struct {
	Listen             string  `toml:"listen"`
	HealthListen       string  `toml:"health_listen"`
	RegistryListen     string  `toml:"registry_listen"`
	Upstream           string  `toml:"upstream"`
	KeysFile           string  `toml:"keys_file"`
	PepperEnv          string  `toml:"pepper_env"`
	KeyCacheTTLS       int     `toml:"key_cache_ttl_s"`
	PolicyFile         string  `toml:"policy_file"`
	UsageDB            string  `toml:"usage_db"`
	CacheEntries       int     `toml:"cache_entries"`
	CacheTTLS          int     `toml:"cache_ttl_s"`
	HeartbeatMissLimit int     `toml:"heartbeat_miss_limit"`
	RoutePolicy        string  `toml:"route_policy"`
	AffinityLoadFactor float64 `toml:"affinity_load_factor"`
	DrainDeadlineS     int     `toml:"drain_deadline_s"`
	Routes             []Route `toml:"routes"`
}

// Route is one [[gateway.routes]] entry, the same shape as Route in
// openapi/admin.v1.yaml.
type Route struct {
	Model    string        `toml:"model" json:"model"`
	Aliases  []string      `toml:"aliases" json:"aliases,omitempty"`
	Targets  []string      `toml:"targets" json:"targets,omitempty"`
	Backends []Backend     `toml:"backends" json:"backends,omitempty"`
	Cascade  []CascadeStep `toml:"cascade" json:"cascade,omitempty"`
}

type Backend struct {
	ServedModel string  `toml:"served_model" json:"served_model"`
	Weight      float64 `toml:"weight" json:"weight"`
}

type CascadeStep struct {
	Model    string `toml:"model" json:"model"`
	AcceptIf string `toml:"accept_if" json:"accept_if,omitempty"`
}

type Engine struct {
	Role            string      `toml:"role"`
	ModelDir        string      `toml:"model_dir"`
	ModelID         string      `toml:"model_id"`
	HTTPListen      string      `toml:"http_listen"`
	HealthListen    string      `toml:"health_listen"`
	GRPCListen      string      `toml:"grpc_listen"`
	KVListen        string      `toml:"kv_listen"`
	GatewayRegistry string      `toml:"gateway_registry"`
	AdvertiseHost   string      `toml:"advertise_host"`
	KVBlocks        int         `toml:"kv_blocks"`
	BlockSize       int         `toml:"block_size"`
	KVFormat        int         `toml:"kv_format"`
	PrefixCache     string      `toml:"prefix_cache"`
	MaxBatchTokens  int         `toml:"max_batch_tokens"`
	MaxSeqs         int         `toml:"max_seqs"`
	PrefillChunk    int         `toml:"prefill_chunk"`
	QueueCapacity   int         `toml:"queue_capacity"`
	QueueDeadlineMS int         `toml:"queue_deadline_ms"`
	Threads         int         `toml:"threads"`
	Quant           string      `toml:"quant"`
	Speculative     Speculative `toml:"speculative"`
}

type Speculative struct {
	Draft string `toml:"draft"`
	K     int    `toml:"k"`
}

type Durable struct {
	GRPCListen          string `toml:"grpc_listen"`
	HealthListen        string `toml:"health_listen"`
	RaftListen          string `toml:"raft_listen"`
	WALDir              string `toml:"wal_dir"`
	WALMaxBytes         int64  `toml:"wal_max_bytes"`
	VisibilityTimeoutMS int    `toml:"visibility_timeout_ms"`
	DLQAfterAttempts    int    `toml:"dlq_after_attempts"`
	LongPollMS          int    `toml:"long_poll_ms"`
	Raft                Raft   `toml:"raft"`
}

type Raft struct {
	Enabled bool       `toml:"enabled"`
	ID      int        `toml:"id"`
	Peers   []RaftPeer `toml:"peers"`
}

type RaftPeer struct {
	ID      int    `toml:"id"`
	Address string `toml:"address"`
}

type Worker struct {
	Durable                    string   `toml:"durable"`
	HealthListen               string   `toml:"health_listen"`
	TaskQueues                 []string `toml:"task_queues"`
	Python                     string   `toml:"python"`
	MaxConcurrentActivities    int      `toml:"max_concurrent_activities"`
	MaxConcurrentWorkflowTasks int      `toml:"max_concurrent_workflow_tasks"`
	Identity                   string   `toml:"identity"`
}

type Agent struct {
	Durable       string `toml:"durable"`
	HealthListen  string `toml:"health_listen"`
	TaskQueue     string `toml:"task_queue"`
	BaseURL       string `toml:"base_url"`
	Model         string `toml:"model"`
	APIKeyEnv     string `toml:"api_key_env"`
	RAGIndex      string `toml:"rag_index"`
	MaxIterations int    `toml:"max_iterations"`
}

// Default is the configuration with every schema default filled in.
func Default() Config {
	return Config{
		Paths: Paths{Artifacts: "/artifacts"},
		OTel:  OTel{TraceSampleRatio: 1.0},
		Gateway: Gateway{
			Listen: ":8080", HealthListen: ":9464", RegistryListen: ":50060",
			PepperEnv: "TL_GATEWAY_PEPPER", KeyCacheTTLS: 5, CacheEntries: 4096,
			CacheTTLS: 300, HeartbeatMissLimit: 3, RoutePolicy: "affinity",
			AffinityLoadFactor: 1.25, DrainDeadlineS: 30,
		},
		Engine: Engine{
			Role: "unified", HTTPListen: ":8000", HealthListen: ":9464",
			GRPCListen: ":50051", KVListen: ":50052", KVBlocks: 2048, BlockSize: 16,
			KVFormat: 1, PrefixCache: "radix", MaxBatchTokens: 2048, MaxSeqs: 64,
			PrefillChunk: 512, QueueCapacity: 256, QueueDeadlineMS: 30000,
			Quant: "none", Speculative: Speculative{Draft: "none", K: 4},
		},
		Durable: Durable{
			GRPCListen: ":7233", HealthListen: ":9464", RaftListen: ":7234",
			WALMaxBytes: 2147483648, VisibilityTimeoutMS: 30000, DLQAfterAttempts: 5,
			LongPollMS: 30000,
		},
		Worker: Worker{HealthListen: ":9464", MaxConcurrentActivities: 4, MaxConcurrentWorkflowTasks: 16},
		Agent:  Agent{HealthListen: ":9464", TaskQueue: "agent", APIKeyEnv: "TL_API_KEY", MaxIterations: 8},
	}
}

// Load reads path and applies env (os.Environ() in a service's main).
func Load(path string, env []string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("config: %w", err)
	}
	c, err := Parse(data, env)
	if err != nil {
		return Config{}, fmt.Errorf("config %s: %w", path, err)
	}
	return c, nil
}

// Parse decodes a runtime.toml document over the defaults, rejects unknown
// keys outside [ext], applies the TL_<SECTION>__<KEY> overrides in env, and
// validates the enumerations.
func Parse(data []byte, env []string) (Config, error) {
	c := Default()
	md, err := toml.Decode(string(data), &c)
	if err != nil {
		return Config{}, err
	}
	var unknown []string
	for _, k := range md.Undecoded() {
		if len(k) > 0 && k[0] == "ext" {
			continue
		}
		unknown = append(unknown, k.String())
	}
	if len(unknown) > 0 {
		sort.Strings(unknown)
		return Config{}, fmt.Errorf("unknown keys: %s", strings.Join(unknown, ", "))
	}
	if err := ApplyEnv(&c, env); err != nil {
		return Config{}, err
	}
	if err := c.Validate(); err != nil {
		return Config{}, err
	}
	return c, nil
}

// EnvName is the override variable of section.key: "TL_" + SECTION + "__" +
// KEY, both upper-cased ("engine", "kv_blocks" -> TL_ENGINE__KV_BLOCKS).
func EnvName(section, key string) string {
	return "TL_" + strings.ToUpper(section) + "__" + strings.ToUpper(key)
}

// ApplyEnv applies every TL_<SECTION>__<KEY>=value in env to c. Variables
// without the double underscore (TL_API_KEY, TL_GATEWAY_PEPPER) are not
// overrides and are ignored. An override naming an unknown section or key,
// a table or array, or a value that does not parse as the key's type is an
// error that names the variable.
func ApplyEnv(c *Config, env []string) error {
	root := reflect.ValueOf(c).Elem()
	// Sort for a deterministic error when several variables are wrong.
	vars := append([]string(nil), env...)
	sort.Strings(vars)
	for _, kv := range vars {
		name, val, found := strings.Cut(kv, "=")
		if !found || !strings.HasPrefix(name, "TL_") {
			continue
		}
		section, key, ok := strings.Cut(strings.TrimPrefix(name, "TL_"), "__")
		if !ok {
			continue // TL_API_KEY and friends: not an override
		}
		sec, ok := field(root, section)
		if !ok || sec.Kind() != reflect.Struct {
			return fmt.Errorf("%s: no section [%s] in runtime.toml", name, strings.ToLower(section))
		}
		f, ok := field(sec, key)
		if !ok {
			return fmt.Errorf("%s: no key %s in [%s]", name, strings.ToLower(key), strings.ToLower(section))
		}
		if err := setScalar(f, val); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
	}
	return nil
}

// field finds the struct field whose toml tag, upper-cased, equals upper.
func field(v reflect.Value, upper string) (reflect.Value, bool) {
	t := v.Type()
	if t.Kind() != reflect.Struct {
		return reflect.Value{}, false
	}
	for i := 0; i < t.NumField(); i++ {
		tag, _, _ := strings.Cut(t.Field(i).Tag.Get("toml"), ",")
		if tag != "" && strings.ToUpper(tag) == upper {
			return v.Field(i), true
		}
	}
	return reflect.Value{}, false
}

// setScalar parses s as f's type. Tables and arrays are not overridable.
func setScalar(f reflect.Value, s string) error {
	switch f.Kind() {
	case reflect.String:
		f.SetString(s)
	case reflect.Int, reflect.Int64:
		n, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
		if err != nil {
			return fmt.Errorf("%q is not an integer", s)
		}
		f.SetInt(n)
	case reflect.Float64:
		x, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
		if err != nil {
			return fmt.Errorf("%q is not a number", s)
		}
		f.SetFloat(x)
	case reflect.Bool:
		b, err := strconv.ParseBool(strings.TrimSpace(s))
		if err != nil {
			return fmt.Errorf("%q is not true or false", s)
		}
		f.SetBool(b)
	default:
		return fmt.Errorf("a %s is not overridable from the environment (only scalars are)", f.Kind())
	}
	return nil
}

// Validate checks the enumerations and ranges of runtime.schema.json that a
// typed decode cannot.
func (c Config) Validate() error {
	oneOf := func(what, v string, allowed ...string) error {
		for _, a := range allowed {
			if v == a {
				return nil
			}
		}
		return fmt.Errorf("%s = %q, want one of %s", what, v, strings.Join(allowed, ", "))
	}
	if err := oneOf("gateway.route_policy", c.Gateway.RoutePolicy, "weighted", "least_outstanding", "affinity"); err != nil {
		return err
	}
	if err := oneOf("engine.role", c.Engine.Role, "unified", "prefill", "decode"); err != nil {
		return err
	}
	if err := oneOf("engine.prefix_cache", c.Engine.PrefixCache, "none", "hash", "radix"); err != nil {
		return err
	}
	if c.Gateway.KeyCacheTTLS < 0 || c.Gateway.KeyCacheTTLS > 5 {
		return fmt.Errorf("gateway.key_cache_ttl_s = %d, want 0 to 5", c.Gateway.KeyCacheTTLS)
	}
	if c.Gateway.AffinityLoadFactor < 1 {
		return fmt.Errorf("gateway.affinity_load_factor = %v, want at least 1", c.Gateway.AffinityLoadFactor)
	}
	if c.OTel.TraceSampleRatio < 0 || c.OTel.TraceSampleRatio > 1 {
		return fmt.Errorf("otel.trace_sample_ratio = %v, want 0 to 1", c.OTel.TraceSampleRatio)
	}
	for _, r := range c.Gateway.Routes {
		if r.Model == "" {
			return fmt.Errorf("gateway.routes: a route without model")
		}
	}
	return nil
}
