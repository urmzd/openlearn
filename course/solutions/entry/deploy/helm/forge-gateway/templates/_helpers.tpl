{{- define "gateway.labels" -}}
app.kubernetes.io/name: {{ .Release.Name }}
app.kubernetes.io/component: gateway
{{- end -}}
