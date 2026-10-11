# Runbook: engine crashloop

Owner: the engine (`<system>-engine`). Written after drill `ops.00`.
Use it when the engine pod restarts over and over and completions fail.

## Symptoms

- Clients get `503` with `"code": "no_capacity"` from `POST /v1/completions`
  through the gateway, before any token.
- `GET /readyz` on the gateway health port answers `503`: the gateway cannot
  reach a healthy engine. `GET /healthz` on the gateway still answers `200`,
  so the gateway itself is fine.
- `kubectl get pods -n <system>` shows the engine pod with `STATUS`
  `CrashLoopBackOff` or `Error`, `READY 0/1`, and a `RESTARTS` count that
  keeps growing. The gap between restarts doubles each time (10 s, 20 s,
  40 s, up to 5 min).

## Diagnosis

Work from the outside in. Each step either names the cause or tells you
where to look next.

1. Confirm which workload is unhealthy:

   ```bash
   kubectl get deploy,pods -n <system> -o wide
   ```

   The gateway pod is `1/1 Running`; the engine pod is not. The fault is in
   the engine, not in the gateway or the network between them.

2. Read why the last container exited, using the engine pod's name from
   step 1:

   ```bash
   kubectl describe pod -n <system> <engine-pod>
   ```

   Look at `Last State: Terminated` (`Reason: Error`, `Exit Code: 1`) and at
   the `Events` (`Back-off restarting failed container`). Exit code 1 at
   startup means the process chose to exit, so its own log says why. Exit
   code 137 would mean it was killed (out of memory), which is a different
   runbook.

3. Read the log of the crashed run, not the current one:

   ```bash
   kubectl logs -n <system> deploy/<system>-engine --previous
   ```

   The engine prints the reason on stderr before it exits, for example
   `cannot read /missing/config.json: No such file or directory`.

4. Compare the running spec with what the chart renders:

   ```bash
   kubectl get deploy <system>-engine -n <system> -o jsonpath='{.spec.template.spec.containers[0].args}'
   helm get manifest <system>-engine -n <system> | grep -A6 'args:'
   kubectl rollout history deploy/<system>-engine -n <system>
   ```

   If the live args differ from the chart, someone changed the Deployment
   outside Helm. The rollout history shows the revision that introduced the
   bad `--model-dir`.

## Mitigation

Restore service first, then fix the cause.

1. Roll the Deployment back to the last revision that worked:

   ```bash
   kubectl rollout undo deploy/<system>-engine -n <system>
   kubectl rollout status deploy/<system>-engine -n <system> --timeout=120s
   ```

2. Check that traffic flows again through the gateway NodePort:

   ```bash
   curl -sS -N http://127.0.0.1:30080/v1/completions \
     -H "Authorization: Bearer $TL_API_KEY" -H 'Content-Type: application/json' \
     -d '{"model": "tracer", "prompt": "Once", "max_tokens": 8, "stream": true}'
   ```

   You see eight `data:` chunks and `data: [DONE]`.

3. Make the fix durable. The chart is the source of truth, so re-apply it:

   ```bash
   helm upgrade --install <system>-engine deploy/helm/<system>-engine -n <system>
   ```

   Never fix the model path with `kubectl edit`: the next `helm upgrade`
   would undo the edit.

## Verification

- `kubectl get pods -n <system>`: every pod `Running` and `READY 1/1`.
- `curl -s -o /dev/null -w '%{http_code}' <gateway health>/readyz` prints `200`.
- `ol drill end` reports the resolve check green.

## Escalation

If the previous log shows a different error (a malformed `config.json`, a
safetensors header the engine rejects, an ABI version mismatch), the model
artifact itself is bad. Roll back as above, then rebuild the model directory
with `{tinyllm} train bigram` and redeploy. If rollback does not help, the
image is at fault: redeploy the last image tag that passed `ol milestone MS-P1`.
