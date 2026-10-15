// Package builtin holds the course packs compiled into the binary, so the app
// has something to teach before the user has imported anything.
//
// The content directory is filled by cmd/bundle (`make content`) and is not
// tracked: it is a copy of packs that live elsewhere in the repository. Only
// its .gitkeep is committed, because //go:embed fails on a missing directory.
package builtin

import (
	"embed"
	"io/fs"
)

//go:embed all:content
var content embed.FS

// FS returns the built-in packs, one directory per pack id. It is empty in a
// build that skipped `make content`.
func FS() fs.FS {
	sub, err := fs.Sub(content, "content")
	if err != nil {
		panic(err) // the directory is embedded above, so this cannot fail
	}
	return sub
}
