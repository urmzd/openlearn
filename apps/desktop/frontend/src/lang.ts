// CodeMirror language support, picked by language name or by file extension.

import { javascript } from "@codemirror/lang-javascript";
import { python } from "@codemirror/lang-python";
import { go } from "@codemirror/lang-go";
import { java } from "@codemirror/lang-java";
import { rust } from "@codemirror/lang-rust";
import type { Extension } from "@codemirror/state";

export function langExtension(lang: string): Extension[] {
  switch (lang) {
    case "python":
    case "py":
      return [python()];
    case "typescript":
    case "ts":
      return [javascript({ typescript: true })];
    case "javascript":
    case "js":
      return [javascript()];
    case "java":
      return [java()];
    case "rust":
    case "rs":
      return [rust()];
    case "go":
      return [go()];
    default:
      // No grammar is better than the wrong one: C, Zig, and plain text files
      // get an editor without highlighting rather than Go's keywords.
      return [];
  }
}

export function langForFile(path: string): Extension[] {
  const ext = path.slice(path.lastIndexOf(".") + 1).toLowerCase();
  return langExtension(ext);
}
