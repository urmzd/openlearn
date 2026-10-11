// Renders one course document.
//
// A pack is markdown written for GitHub, so this renders GitHub's dialect:
// tables, raw HTML, fenced code, LaTeX math, and mermaid diagrams. Links between
// documents stay inside the app; everything else opens in the browser.
//
// A pack can come from anywhere, so its HTML is sanitized before anything else
// touches it. Math and highlighting run after that, on markup this file made.

import { useEffect, useRef, useState, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeRaw from "rehype-raw";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import { BrowserOpenURL } from "../wailsjs/runtime/runtime";
import { isDoc, isImage, resolveLink } from "./paths";
import "katex/dist/katex.min.css";
import "highlight.js/styles/github-dark.css";
import "./markdown.css";

const schema = {
  ...defaultSchema,
  attributes: {
    ...defaultSchema.attributes,
    // remark-math marks math with these classes; the default schema only lets
    // language-* through on code.
    code: [["className", /^language-./, "math-inline", "math-display"]],
  },
};

interface Props {
  /** Pack-relative path of the document, for resolving its relative links. */
  path: string;
  markdown: string;
  /** Where links to files the app does not render (code, configs) point. */
  repo?: string;
  onOpenDoc: (path: string) => void;
}

function textOf(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (node && typeof node === "object" && "props" in node) {
    return textOf((node as { props: { children?: ReactNode } }).props.children);
  }
  return "";
}

let mermaidSeq = 0;

/** One mermaid diagram. The library is large, so it loads on first use. */
function Mermaid({ chart }: { chart: string }) {
  const [svg, setSvg] = useState("");
  const [failed, setFailed] = useState(false);
  const id = useRef(`mermaid-${++mermaidSeq}`);

  useEffect(() => {
    let live = true;
    import("mermaid")
      .then(async ({ default: mermaid }) => {
        mermaid.initialize({ startOnLoad: false, theme: "dark", securityLevel: "strict" });
        const out = await mermaid.render(id.current, chart);
        if (live) setSvg(out.svg);
      })
      .catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, [chart]);

  // A diagram that does not parse is still information: show its source.
  if (failed) return <pre className="md-pre">{chart}</pre>;
  if (!svg) return <div className="md-mermaid faint">Rendering diagram…</div>;
  return <div className="md-mermaid" dangerouslySetInnerHTML={{ __html: svg }} />;
}

export function Markdown({ path, markdown, repo, onOpenDoc }: Props) {
  const components: Components = {
    a({ href, children }) {
      const target = href ? resolveLink(path, href) : null;
      const follow = (e: React.MouseEvent) => {
        e.preventDefault();
        if (!target) return;
        if (target.kind === "external") return BrowserOpenURL(target.path);
        if (target.kind === "anchor") {
          return document.getElementById(target.hash)?.scrollIntoView({ behavior: "smooth" });
        }
        if (isDoc(target.path)) return onOpenDoc(target.path);
        // A link to a directory means its README, as it does on GitHub.
        if (!/\.[a-z0-9]+$/i.test(target.path)) {
          return onOpenDoc(target.path ? `${target.path}/README.md` : "README.md");
        }
        if (repo) BrowserOpenURL(`${repo.replace(/\/$/, "")}/blob/main/${target.path}`);
      };
      const outside = target?.kind === "external" || (target?.kind === "pack" && !isDoc(target.path) && /\.[a-z0-9]+$/i.test(target.path));
      return (
        <a href={href} onClick={follow} className={outside ? "md-external" : undefined}>
          {children}
        </a>
      );
    },
    img({ src, alt, width, height }) {
      const target = src ? resolveLink(path, src) : null;
      const url = target?.kind === "pack" && isImage(target.path) ? `/pack/${target.path}` : src;
      return <img src={url} alt={alt ?? ""} width={width} height={height} loading="lazy" />;
    },
    pre({ children }) {
      const child = Array.isArray(children) ? children[0] : children;
      const cls = (child as { props?: { className?: string } })?.props?.className ?? "";
      if (cls.includes("language-mermaid")) return <Mermaid chart={textOf(children).trim()} />;
      return <pre className="md-pre">{children}</pre>;
    },
    table({ children }) {
      return (
        <div className="md-table">
          <table>{children}</table>
        </div>
      );
    },
  };

  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[
          rehypeRaw,
          [rehypeSanitize, schema],
          rehypeKatex,
          [rehypeHighlight, { detect: false, ignoreMissing: true }],
        ]}
        components={components}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  );
}
