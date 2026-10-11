package domain

// Skill areas are a fixed taxonomy. Generated rubrics must map every criterion
// onto one of these IDs, which is what makes scores comparable across sessions:
// per-problem rubric wording changes every run, area IDs do not.

// Area identifies a skill dimension a session can score.
type Area string

// System design areas.
const (
	AreaRequirements  Area = "requirements"
	AreaAPIDesign     Area = "api_design"
	AreaDataModeling  Area = "data_modeling"
	AreaStorageChoice Area = "storage_choice"
	AreaScaling       Area = "scaling"
	AreaCaching       Area = "caching"
	AreaConsistency   Area = "consistency"
	AreaReliability   Area = "reliability"
	AreaObservability Area = "observability"
	AreaSecurity      Area = "security"
	AreaCost          Area = "cost"
)

// Coding areas.
const (
	AreaDecomposition Area = "decomposition"
	AreaCorrectness   Area = "correctness"
	AreaEdgeCases     Area = "edge_cases"
	AreaComplexity    Area = "complexity"
	AreaDataStructure Area = "data_structure"
	AreaClarity       Area = "clarity"
	AreaTesting       Area = "testing"
	AreaDebugging     Area = "debugging"
)

// Shared across both modes.
const (
	AreaCommunication Area = "communication"
	AreaTradeoffs     Area = "tradeoffs"
	AreaPacing        Area = "pacing"
)

// AreaInfo is the human-facing description of a skill area.
type AreaInfo struct {
	ID    Area   `json:"id"`
	Label string `json:"label"`
	Blurb string `json:"blurb"`
}

var systemDesignAreas = []AreaInfo{
	{AreaRequirements, "Requirements & scoping", "Pins down functional needs, scale numbers, and what is explicitly out of scope before designing."},
	{AreaAPIDesign, "API & interface design", "Defines clean entry points: endpoints, payloads, idempotency, versioning."},
	{AreaDataModeling, "Data modeling", "Entities, relationships, access patterns, and how they drive the schema."},
	{AreaStorageChoice, "Storage choice", "Picks stores that fit the access pattern and justifies the pick against alternatives."},
	{AreaScaling, "Scaling & throughput", "Partitioning, replication, load balancing, and back-of-envelope capacity math."},
	{AreaCaching, "Caching", "What to cache, where, invalidation strategy, and the cost of getting it wrong."},
	{AreaConsistency, "Consistency & correctness", "Consistency model, transactions, ordering, exactly-once vs at-least-once."},
	{AreaReliability, "Reliability & failure", "Failure modes, retries, backpressure, degradation, blast radius."},
	{AreaObservability, "Observability", "Metrics, logs, traces, and the alerts that would actually page someone."},
	{AreaSecurity, "Security & privacy", "AuthN/AuthZ, data protection, tenancy isolation, abuse resistance."},
	{AreaCost, "Cost & pragmatism", "Right-sizes the design; avoids gold-plating; names the cheap first version."},
	{AreaTradeoffs, "Tradeoff reasoning", "Names alternatives and argues the choice instead of asserting it."},
	{AreaCommunication, "Communication", "Structured narration, clear naming, a board someone else could read."},
	{AreaPacing, "Pacing", "Spends time proportional to what matters; finishes the shape before polishing."},
}

var codingAreas = []AreaInfo{
	{AreaDecomposition, "Problem decomposition", "Breaks the problem into named pieces before writing the body."},
	{AreaCorrectness, "Correctness", "The core algorithm actually solves the stated problem."},
	{AreaEdgeCases, "Edge cases", "Empty, single, duplicate, overflow, and boundary inputs are handled."},
	{AreaComplexity, "Complexity analysis", "Knows the time and space cost and whether it meets the constraint."},
	{AreaDataStructure, "Data structure choice", "Picks structures that make the operations cheap, and says why."},
	{AreaClarity, "Code clarity", "Readable names, small functions, no dead scaffolding."},
	{AreaTesting, "Testing", "Exercises the solution with cases that could actually fail."},
	{AreaDebugging, "Debugging & iteration", "Recovers from a wrong first attempt without flailing."},
	{AreaTradeoffs, "Tradeoff reasoning", "Names alternatives and argues the choice instead of asserting it."},
	{AreaCommunication, "Communication", "Explains intent as the code lands; comments earn their place."},
	{AreaPacing, "Pacing", "Gets to a working solution before optimizing it."},
}

// AreasFor returns the skill taxonomy that applies to a mode.
func AreasFor(mode Mode) []AreaInfo {
	if mode == ModeCoding {
		return codingAreas
	}
	return systemDesignAreas
}

// AreaIDsFor returns just the area IDs valid for a mode, for schema enums.
func AreaIDsFor(mode Mode) []string {
	infos := AreasFor(mode)
	ids := make([]string, len(infos))
	for i, a := range infos {
		ids[i] = string(a.ID)
	}
	return ids
}

// LabelFor returns the human label for an area, falling back to the raw ID.
func LabelFor(mode Mode, id Area) string {
	for _, a := range AreasFor(mode) {
		if a.ID == id {
			return a.Label
		}
	}
	return string(id)
}

// ValidArea reports whether id belongs to the taxonomy for mode.
func ValidArea(mode Mode, id Area) bool {
	for _, a := range AreasFor(mode) {
		if a.ID == id {
			return true
		}
	}
	return false
}
