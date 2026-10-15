# Paths

A path is a reading order cut across the tracks. The [course](course/) and its pass paths build one system: their stages carry a check column that `ol check`, `ol milestone`, and `ol drill` grade. The tracks stay where they are and own their content; a path only decides **which chapters, in what order, and what "done" means**. Nothing is copied, so a chapter fixed in its track is fixed in every path that uses it.

| Path | For | Stages |
|------|-----|--------|
| [course](course/) | The end-to-end course: build your own LLM system in Python, Rust, Go, and C, pass by pass | 21 |
| [course-p00-setup](course-p00-setup/) to [course-p12-multimodal](course-p12-multimodal/) | One path per pass, each ending at its milestone gate | per pass |

## Reading a path

```bash
practice/bin/ol learn                                  # every path, and your progress
practice/bin/ol learn course                           # stages, with done-when criteria
practice/bin/ol learn course next                      # read the first unfinished stage
practice/bin/ol learn course-p01-tracer 2              # read one stage
practice/bin/ol learn course-p01-tracer --done 2       # mark it done once its check passes
```

A path's own stages are addressed as `<n>`, an included path's as `<path>:<n>`. Progress lives in `.scratchpad/learn/` and is never committed. Stages render with [glow](https://github.com/charmbracelet/glow) when it is installed, and as plain markdown through `$PAGER` otherwise.

Each path also builds into its own PDF (`supersource-<path>.pdf`): CI attaches every one to the build artifact and to each GitHub Release, and `./scripts/build-book.sh --path <path>` builds one locally. A path that includes others gets one PDF Part per included path.

## Adding a path

1. Create `content/courses/build-an-llm/paths/<name>/README.md`. Its first `# ` heading is the path's title.
2. Create `content/courses/build-an-llm/paths/<name>/path.tsv`, one row per stage, tab-separated:

   ```text
   stage <TAB> title <TAB> file[,file...] <TAB> done when
   @other-path
   ```

   Files are repo-relative markdown, read in the order listed. A line `@other-path` includes every stage of that path at that point. Lines starting with `#` are comments. The fifth column is `module:<ID>`, `milestone:<ID>`, `drill:<ID>`, or `-` for a manual stage. "Done when" is a checkable outcome (you built, measured, wrote, or explained something), not "read the chapter".
3. Content belongs in a track, even when only one path uses it today. A path directory holds only its manifest, its README, and material that is about the path itself, such as a course pass intro or milestone page.
4. Run `practice/bin/ol learn --verify`. CI runs it too: a missing file, an unknown include, or an include cycle fails the build.
5. Add `outputs/supersource-<name>.pdf` to `sr.yaml` under `packages[].artifacts` so releases attach it.
