// Preserve section URLs, asset URLs, and reading-progress identities across book merges.
import previousPaths from './source-migrations.json' with { type: 'json' };

export function stableSourcePath(file) {
  for (const [current, previous] of previousPaths) {
    if (file === current || file.startsWith(`${current}/`)) return previous + file.slice(current.length);
  }
  return file;
}
