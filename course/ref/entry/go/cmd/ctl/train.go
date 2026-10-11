package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

// The train verb: a training run as the TrainRun workflow (dur.11).
func init() {
	register("train", train)
}

// trainRunResult is the part of TrainRun's result ctl reads.
type trainRunResult struct {
	Steps      int    `json:"steps"`
	Checkpoint string `json:"checkpoint"`
}

// train starts TrainRun on a train spec (formats/train-spec.schema.json),
// with "out" defaulting to runs/<workflow id>.
// With --wait the final line is {workflow_id, status, steps, final_loss,
// ckpt}; without it, {workflow_id, run_id, started, status}.
func train(args []string) error {
	fs := newFlags("train")
	specPath := fs.String("spec", "", "train spec, JSON")
	id := fs.String("id", "", "workflow id")
	queue := fs.String("queue", "default", "task queue")
	wait := fs.Bool("wait", false, "wait for the run and print its final loss")
	addr := durableFlag(fs)
	jsonFlag(fs)
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	if err := exactly("train", pos, 0, "no positional arguments"); err != nil {
		return err
	}
	if *specPath == "" || *id == "" {
		return usagef("train: --spec and --id are required")
	}
	spec, err := os.ReadFile(*specPath)
	if err != nil {
		return usagef("train: --spec: %v", err)
	}
	var obj map[string]any
	if err := json.Unmarshal(spec, &obj); err != nil {
		return usagef("train: --spec %s is not a JSON object: %v", *specPath, err)
	}
	// A spec without "out" trains in runs/<workflow id>: every segment of
	// this run resumes its own checkpoints (ckpt/LATEST), and another run of
	// the same spec under another id starts fresh instead of resuming this
	// one's final checkpoint.
	if _, ok := obj["out"]; !ok {
		obj["out"] = "runs/" + *id
		if spec, err = json.Marshal(obj); err != nil {
			return err
		}
	}
	input, err := json.Marshal(map[string]json.RawMessage{"spec": spec})
	if err != nil {
		return err
	}
	d, err := dial(*addr)
	if err != nil {
		return err
	}
	defer d.Close()
	resp, info, err := d.run("TrainRun", *id, *queue, input, *wait)
	if err != nil {
		return err
	}
	status := statusName(info.GetStatus())
	if !*wait {
		return emit(map[string]any{"workflow_id": *id, "run_id": resp.GetRunId(), "started": resp.GetStarted(), "status": status})
	}
	if err := notCompleted(info); err != nil {
		_ = emit(map[string]any{"workflow_id": *id, "status": status})
		return err
	}
	var res trainRunResult
	if err := json.Unmarshal(info.GetResult(), &res); err != nil {
		return fmt.Errorf("TrainRun result: %w", err)
	}
	loss, err := finalLoss(d, info, res.Checkpoint)
	if err != nil {
		return err
	}
	return emit(map[string]any{
		"workflow_id": *id, "status": status, "steps": res.Steps,
		"final_loss": loss, "ckpt": res.Checkpoint,
	})
}

// finalLoss reads <run dir>/final.json, which `{tinyllm} train --spec`
// writes next to ckpt/ (the checkpoint is <run dir>/ckpt/step-NNNNNN).
func finalLoss(d *durable, info *durablev1.WorkflowInfo, ckpt string) (float64, error) {
	run := path.Dir(path.Dir(ckpt))
	b, err := d.readArtifact(info, path.Join(run, "final.json"))
	if err != nil {
		return 0, err
	}
	var final struct {
		Loss *float64 `json:"loss"`
	}
	if err := json.Unmarshal(b, &final); err != nil || final.Loss == nil {
		return 0, fmt.Errorf("%s/final.json has no loss: %s", run, b)
	}
	return *final.Loss, nil
}
