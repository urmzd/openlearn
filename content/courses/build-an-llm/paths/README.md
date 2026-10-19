# Paths

A path is a reading order cut across the tracks. The [course](course/) and its pass paths walk through one system: each stage names the chapters to read and what you should be able to explain afterwards. The tracks stay where they are and own their content; a path only decides **which chapters, in what order, and what "done" means**. Nothing is copied, so a chapter fixed in its track is fixed in every path that uses it.

| Path | For | Stages |
|------|-----|--------|
| [course](course/) | The end-to-end course: how one LLM system is built in Python, Rust, Go, and C, pass by pass, with the solution to every chapter | 21 |
| [course-p00-setup](course-p00-setup/) to [course-p12-multimodal](course-p12-multimodal/) | One path per pass, each ending with what the system does after that pass | per pass |

## Reading a path

Open a path's `path.tsv` and read its stages in order. Each row names the chapters of one stage and a "done when" statement: the thing you should be able to explain or reproduce once you have read the chapters and their solution walkthroughs. A path's own stages are numbered `<n>`; a line `@<path>` includes every stage of another path at that point.

Each path also builds into its own PDF (`supersource-<path>.pdf`): CI attaches every one to the build artifact and to each GitHub Release, and `./scripts/build-book.sh --path <path>` builds one locally. A path that includes others gets one PDF Part per included path.

## Adding a path

1. Create `content/courses/build-an-llm/paths/<name>/README.md`. Its first `# ` heading is the path's title.
2. Create `content/courses/build-an-llm/paths/<name>/path.tsv`, one row per stage, tab-separated:

   ```text
   stage <TAB> title <TAB> file[,file...] <TAB> done when
   @other-path
   ```

   Files are repo-relative markdown, read in the order listed. A line `@other-path` includes every stage of that path at that point. Lines starting with `#` are comments. "Done when" is a comprehension outcome (you can explain, derive, or reproduce something), not "read the chapter".
3. Content belongs in a track, even when only one path uses it today. A path directory holds only its manifest, its README, and material that is about the path itself, such as a course pass intro or its "what the system does" page.
4. Run `./scripts/build-book.sh --path <name>`. A missing file, an unknown include, or an include cycle fails the build, and CI runs the same assembly.
5. Add `outputs/supersource-<name>.pdf` to `sr.yaml` under `packages[].artifacts` so releases attach it.
