{{- define "hello.labels" -}}
app.kubernetes.io/name: hello
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
