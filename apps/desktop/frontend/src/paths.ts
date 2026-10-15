// Path arithmetic for links inside a course pack. Documents link to each other
// with relative paths, the way they would on GitHub; these resolve them against
// the pack root so the reader can follow them.

/** Normalizes a slash path, resolving "." and "..". Returns null if it climbs
 *  above the root, which no link inside a pack should do. */
export function normalize(path: string): string | null {
  const out: string[] = [];
  for (const part of path.split("/")) {
    if (part === "" || part === ".") continue;
    if (part === "..") {
      if (out.length === 0) return null;
      out.pop();
      continue;
    }
    out.push(part);
  }
  return out.join("/");
}

export function dirname(path: string): string {
  const i = path.lastIndexOf("/");
  return i < 0 ? "" : path.slice(0, i);
}

export interface Target {
  kind: "external" | "anchor" | "pack";
  /** For "pack": the pack-relative path. For the others: the original href. */
  path: string;
  hash: string;
}

/** Classifies a link found in the document at `from`. */
export function resolveLink(from: string, href: string): Target | null {
  if (href.startsWith("#")) return { kind: "anchor", path: href, hash: href.slice(1) };
  if (/^[a-z][a-z0-9+.-]*:/i.test(href) || href.startsWith("//")) {
    return { kind: "external", path: href, hash: "" };
  }
  const [raw, hash = ""] = href.split("#");
  let decoded = raw;
  try {
    decoded = decodeURI(raw);
  } catch {
    /* a malformed escape is left as written */
  }
  const joined = decoded.startsWith("/") ? decoded : `${dirname(from)}/${decoded}`;
  const path = normalize(joined);
  if (path === null) return null;
  return { kind: "pack", path, hash };
}

const IMAGE = /\.(svg|png|jpe?g|gif|webp)$/i;

export const isImage = (path: string) => IMAGE.test(path);
export const isDoc = (path: string) => path.toLowerCase().endsWith(".md");
