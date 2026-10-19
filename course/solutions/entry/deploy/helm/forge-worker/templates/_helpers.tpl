{{- define "worker.labels" -}}
app.kubernetes.io/name: {{ .Release.Name }}
app.kubernetes.io/component: worker
{{- end -}}
{{- define "worker.runtime" -}}
{{- $w := dict "durable" .Values.durableAddress "health_listen" ":9464" "task_queues" .Values.taskQueues "python" .Values.python -}}
{{- toToml (dict "worker" $w "paths" (dict "artifacts" .Values.artifacts.mountPath) "otel" (dict "endpoint" .Values.otel.endpoint)) -}}
{{- end -}}
