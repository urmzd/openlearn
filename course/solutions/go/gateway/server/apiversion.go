package server

import (
    "encoding/json"
    "net/http"
    "strconv"
    "time"
)

const APIVersionHeader = "X-TL-API-Version"
type versionError struct { Error struct { Message string `json:"message"`; Type string `json:"type"`; Param *string `json:"param"`; Code string `json:"code"` } `json:"error"` }
type APIVersionConfig struct { Default string; DeprecatedAt time.Time; Sunset time.Time; Now func() time.Time }

// VersionedAPI applies explicit v1/v2 selection at the HTTP boundary, echoes
// the selected version, and rejects v1 at the configured sunset.
func VersionedAPI(cfg APIVersionConfig, next http.Handler) http.Handler {
    if cfg.Default=="" { cfg.Default="1" }; if cfg.Now==nil { cfg.Now=time.Now }
    return http.HandlerFunc(func(w http.ResponseWriter,r *http.Request) {
        v:=r.Header.Get(APIVersionHeader); if v=="" { v=cfg.Default }
        if v!="1"&&v!="2" { writeVersionError(w,http.StatusBadRequest,"unsupported_api_version"); return }
        w.Header().Set(APIVersionHeader,v)
        if v=="1" {
            if !cfg.DeprecatedAt.IsZero() { w.Header().Set("Deprecation","@"+strconv.FormatInt(cfg.DeprecatedAt.Unix(),10)) }
            if !cfg.Sunset.IsZero() { w.Header().Set("Sunset",cfg.Sunset.UTC().Format(http.TimeFormat)); if !cfg.Now().Before(cfg.Sunset) { writeVersionError(w,http.StatusGone,"api_version_sunset"); return } }
        }
        r.Header.Set(APIVersionHeader,v) // the usage meter records this selected version
        next.ServeHTTP(w,r)
    })

}
func writeVersionError(w http.ResponseWriter,status int,code string) {
        w.Header().Set("Content-Type","application/json");w.WriteHeader(status);var e versionError;e.Error.Message="API version is unavailable";e.Error.Type="invalid_request_error";e.Error.Code=code;_ = json.NewEncoder(w).Encode(e) 
}
