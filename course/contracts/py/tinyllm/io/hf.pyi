# contracts/py/tinyllm/io/hf.pyi (L7.9): download files from the Hugging Face Hub
# chapter: content/courses/build-an-llm/p07-modern-block/09-llama-family-model.md
#
# The Hub serves every file of a repository at
#
#     {endpoint}/{repo_id}/resolve/{revision}/{filename}
#
# endpoint is the argument, else the HF_ENDPOINT environment variable, else
# https://huggingface.co. Large (LFS) files answer with a redirect to a CDN;
# the first response carries X-Linked-Etag, the file's sha256 in quotes, and
# X-Linked-Size. A request may carry "Range: bytes=<start>-" to resume: the
# server answers 206 with the rest, or 200 with the whole file when it
# ignores ranges. HF_TOKEN, when set, is sent as "Authorization: Bearer".
#
# Files land in <cache_dir>/<repo_id with "/" as "--">/<revision>/<filename>.
# A download writes <filename>.part first and renames it into place only after
# its size and sha256 check out, so a crash or a bad byte never leaves a
# truncated or corrupt file under the final name.
from typing import Mapping, Optional, Sequence

def hf_download(
    repo_id: str,
    filenames: Sequence[str],
    cache_dir: str,
    revision: str = "main",
    endpoint: Optional[str] = None,
    sha256: Optional[Mapping[str, str]] = None,
    timeout: float = 60.0,
) -> str:
    """Download each filename (skipping a final file that is already there
    and, when its sha256 is known, matches) and return the directory that
    holds them. Resumes from an existing .part file with a Range request
    (hashing the bytes already there first). The expected sha256 of a file
    is sha256[filename] when given, else the X-Linked-Etag the server sent
    for it (a 64-hex-digit value), else unchecked; the expected size is
    X-Linked-Size or the response's total length. ValueError for a sha256
    mismatch or more bytes than expected (the .part file is deleted), a
    repo_id that is not "owner/name", or a filename with ".." or a leading
    "/"; OSError (urllib's HTTPError and URLError included) for an HTTP or
    network failure or a body shorter than expected (the .part file is kept
    for a resume)."""
