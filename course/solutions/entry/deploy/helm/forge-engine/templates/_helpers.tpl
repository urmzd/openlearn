{{- define "engine.labels" -}}
app.kubernetes.io/name: {{ .Release.Name }}
app.kubernetes.io/component: engine
tinyllm.role: {{ .Values.role }}
{{- end -}}
