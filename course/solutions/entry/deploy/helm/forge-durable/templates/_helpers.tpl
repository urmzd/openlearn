{{- define "durable.labels" -}}
app.kubernetes.io/name: {{ .Release.Name }}
app.kubernetes.io/component: durable
{{- end -}}
{{- /* runtime.toml: Helm reads YAML numbers as floats, and toToml would
       write 5.0 where runtime.schema.json wants an integer, so every
       integer key is cast with int64. */ -}}
{{- define "durable.runtime" -}}
{{- $r := default dict (.Values.runtime).durable -}}
{{- $d := dict "grpc_listen" (printf ":%d" (int .Values.service.grpcPort)) "health_listen" (printf ":%d" (int .Values.service.healthPort)) "wal_dir" "/var/lib/durable/wal" "wal_max_bytes" (int64 .Values.walMaxBytes) -}}
{{- range $k := list "visibility_timeout_ms" "dlq_after_attempts" "long_poll_ms" -}}
{{- if hasKey $r $k }}{{ $_ := set $d $k (int64 (get $r $k)) }}{{ end -}}
{{- end -}}
{{- toToml (dict "durable" $d "otel" (dict "endpoint" $.Values.otel.endpoint)) -}}
{{- end -}}
