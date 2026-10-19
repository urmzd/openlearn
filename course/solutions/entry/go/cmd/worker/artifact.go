package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"tinyllm/activities"
	"tinyllm/durable/activity"
	"tinyllm/durable/worker"
	"tinyllm/durable/workflow"
)

// actReadArtifact is the activity behind ReadArtifact.
const actReadArtifact = "artifact.read"

var readOptions = workflow.ActivityOptions{StartToClose: 30 * time.Second}

// readArtifact is the ReadArtifact workflow: input a JSON string, a path
// relative to the artifacts root; result the file's bytes. Activity results
// are paths (spec/subprocess-activity.md), so a client that wants what a
// path holds (ctl's RUN object and final loss) asks a worker, which shares
// the artifacts root with the activities that wrote it.
func readArtifact(ctx workflow.Context, in []byte) ([]byte, error) {
	return workflow.ExecuteActivity[[]byte](ctx, actReadArtifact, in, readOptions).Get(ctx)
}

// artifactReader reads one file under root. A path that is absolute, leaves
// the root, or names a file larger than an activity result can be fails
// without a retry.
func artifactReader(root string) worker.ActivityFunc {
	return func(_ context.Context, in []byte) ([]byte, error) {
		var rel string
		if err := json.Unmarshal(in, &rel); err != nil {
			return nil, activity.NewNonRetryable("BadInput", "input is not a JSON string: "+err.Error())
		}
		clean := path.Clean(rel)
		if rel == "" || path.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, "../") {
			return nil, activity.NewNonRetryable("BadInput", fmt.Sprintf("%q is not a path under the artifacts root", rel))
		}
		p := filepath.Join(root, filepath.FromSlash(clean))
		st, err := os.Stat(p)
		if err != nil {
			return nil, activity.NewNonRetryable("NotFound", err.Error())
		}
		if st.Size() > activities.MaxResult {
			return nil, activity.NewNonRetryable("ResultTooLarge", fmt.Sprintf("%s is %d bytes (max %d)", rel, st.Size(), activities.MaxResult))
		}
		return os.ReadFile(p)
	}
}
