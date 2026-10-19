# Reading Paths

The chapters of this book are arranged by topic: tokenizers in one chapter, attention in another, serving in a third. That is the right arrangement for reference, but it is not always the right order for learning. A reading path is a second arrangement laid over the first. It cuts across the chapters, and across the other books in the library, and picks out which sections to read, in what order, and what you should be able to explain when you finish each stage.

A path owns no content of its own beyond its opening page and a short account of what the system can do at its end. The sections stay in the chapters and books that own them, so a correction made in a chapter is made in every path that uses it. Nothing is copied.

## The course and its passes

The main path is [the course](course/README.md): one LLM system, built end to end in Python, Rust, Go, and C, grown in thirteen passes. Each pass is itself a path, and each ends with a page describing what the finished system does at that point.

- [Pass 0: Setup](course-p00-setup/README.md): the toolchain, the shell, and a CI gate that every later pass relies on.
- [Pass 1: The tracer](course-p01-tracer/README.md): a thin system through every layer, from a counted bigram to one trace in Jaeger.
- [Pass 2: Foundations](course-p02-foundations/README.md): the mathematics and the autograd that retrain the tracer's model by gradient descent.
- [Pass 3: Tokens and data](course-p03-tokens-and-data/README.md): tokenizers that match GPT-2 and SmolLM2, a deterministic corpus pipeline, and the statistical language models.
- [Pass 4: Sequence models](course-p04-sequence-models/README.md): recurrent language models, encoder-decoders, and attention's first appearance.
- [Pass 5: The transformer](course-p05-transformer/README.md): the 2017 transformer, its pretraining objectives, and the modern block that loads SmolLM2.
- [Pass 6: Inference and kernels](course-p06-inference-and-kernels/README.md): the Python inference stack, with optional C kernels.
- [Pass 7: The serving platform](course-p07-serving-platform/README.md): the Rust engine, the production gateway, and the platform on Kubernetes with its SLOs.
- [Pass 8: Durable execution](course-p08-durable/README.md): the data pipeline and training run as durable workflows.
- [Pass 9: Capstone training](course-p09-capstone-training/README.md): a ten-million-parameter model trained, evaluated, released, and served.
- [Pass 10: Agents](course-p10-agents/README.md): agents on the engine's tool calls, retrieval, evaluation, and usage policy.
- [Pass 11: Operating the system](course-p11-operate/README.md): v1.0.0, migrations, incident drills, security, and review.
- [Pass 12: Multimodal](course-p12-multimodal/README.md): images and audio, vision encoders, and Whisper.

## How a path is read

Every path has a manifest, `path.tsv`, beside its opening page. Each row of the manifest is one stage: a stage number, a title, the files to read in order, and a "done when" statement that names what you should be able to explain or reproduce once you have read those sections and their solution walkthroughs. A line of the form `@<path>` includes every stage of another path at that point, which is how [the course](course/README.md) strings its thirteen passes together. Each pass's opening page turns its manifest into a reading list, so in practice you read the opening page and follow its links.

Each path also builds into its own PDF, `supersource-<path>.pdf`. CI attaches every one to the build artifact and to each GitHub Release, and `./scripts/build-book.sh --path <path>` builds one locally. A path that includes others gets one PDF part per included path.

## Adding a path

Adding a path takes five steps, and the build checks them for you:

1. Create `content/courses/build-an-llm/paths/<name>/README.md`. Its first `# ` heading is the path's title.
2. Create `content/courses/build-an-llm/paths/<name>/path.tsv`, one row per stage, tab-separated:

   ```text
   stage <TAB> title <TAB> file[,file...] <TAB> done when
   @other-path
   ```

   Files are repository-relative Markdown, read in the order listed. A line `@other-path` includes every stage of that path at that point. Lines starting with `#` are comments. "Done when" is a comprehension outcome (you can explain, derive, or reproduce something), not "read the chapter".
3. Put content in a chapter, even when only one path uses it today. A path directory holds only its manifest, its opening page, and material that is about the path itself, such as a pass's opening or its "what the system does" page.
4. Run `./scripts/build-book.sh --path <name>`. A missing file, an unknown include, or an include cycle fails the build, and CI runs the same assembly.
5. Add `outputs/supersource-<name>.pdf` to `sr.yaml` under `packages[].artifacts` so releases attach it.

The third rule is the one that keeps the library coherent. If a path ever needs a section that no chapter has, write the section in the chapter where it belongs and link to it from the path.
