<!-- ol:module dur.01 -->
# Append-only segmented event log

## Overview

| | |
|---|---|
| **Module** | `dur.01` · build · Go · Pass 8 · 4 to 6 h |
| **You build** | `go/durable/log/record.go`: `CRC32C`, `AppendRecord`, `ParseRecord`, `EncodeBatch`, `DecodeBatch`; `go/durable/log/log.go`: `Open`, `Append`, `Read`, `Version`, `Streams`, `LastSeq`, `Size`, `Close` |
| **Contract** | the bytes on disk are [`formats/wal.md`](../../course/contracts/formats/wal.md) (records, segments, recovery, quota); the Go API is section 4 of this chapter, held by the course tests |
| **Tests** | `course/tests/go/dur_01/` (what they check: section 4) · your own tests in `go/durable/log/log_learner_test.go`, rung R4, graded by mutation (threshold 0.80, every pitfall's mutant required) |
| **Needs** | reading: [`lang.06` Go](../../software-craftsmanship/12-language-and-tool-primers/06-go.md), [`lang.10` Protocol Buffers and gRPC](../../software-craftsmanship/12-language-and-tool-primers/10-protocol-buffers-and-grpc.md), [`M06.3` hashing](../../math/06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) |
| **Used by** | `dur.02` keeps every workflow run's history in it · `dur.03` keeps every queue transition in it |
| **Milestone** | MS-durable |
| **Optional depth** | Kleppmann, *Designing Data-Intensive Applications*, ch. 3 (logs) and 7; Pillai et al., *All File Systems Are Not Created Equal* (OSDI 2014); the Castagnoli paper, *Optimization of Cyclic Redundancy-Check Codes with 24 and 32 Parity Bits* (1993) |

## Key Takeaways

- A log record is framed as length, CRC-32C, sequence number, payload; the CRC covers the sequence number too, so a record copied to the wrong place is caught (`TestRecordHandExample`, `TestParseRecordRejectsDamage`).
- Recovery never guesses: damage in the **last** segment is a torn tail from a crash mid-append and is truncated; damage anywhere **before** it is lost acknowledged data and stops the server (`TestTornTailTruncatedAtEveryOffset`, `TestCorruptMiddleSegmentRefused`).
- An append is acknowledged only after `write` **and** `fsync`; a failure in between cuts the bytes back off, or a record nobody acknowledged comes back after a restart (`TestSyncBeforeAck`, `TestFailedAppendLeavesNoTrace`).
- Streams give optimistic concurrency: `Append(stream, expected, ...)` succeeds only if the stream is still at `expected`, so two writers can never both write version $v+1$ (`TestConcurrentAppendersConflict`).
- SIGKILL the writer at 30 random instants and every acknowledged version is still there (`TestKillLoopAckedSurvive`).

## How to work this chapter

```bash
ol start dur.01          # writes go/durable/log/{record,log}.go with stub bodies
ol tests dur.01          # read the test catalog first
ol check dur.01          # exit code is the verdict
ol diff  dur.01          # after passing: your code against the reference
```

Write `record.go` first and check it on the worked example of section 3, then `Open` and `Append` without segments, then segment rolling, then recovery.

---

## 1. Why now

Your platform is about to run work that takes hours: building the corpus, training, evaluating, releasing. Today that work lives in a process's memory, and a `kill -9`, a deploy, or a node drain loses all of it; the only recovery is to start over. The durable engine of this part fixes that by writing down every step before it is considered done, and the thing it writes to is this module: an append-only log on disk. Everything later (workflow histories in `dur.02`, queue leases in `dur.03`, Raft replication in `dur.10`) is a stream of records in it, so its two promises carry the whole engine: nothing acknowledged is ever lost, and nothing half-written is ever believed.

## 2. Principles

### 2.1 A record, and why it carries a checksum

| Symbol | Meaning | Type |
|---|---|---|
| $L$ | payload length in bytes | `u32` |
| $s$ | sequence number of the record, 1 for the first record ever | `u64` |
| $P$ | payload bytes | `[]byte`, length $L$ |
| $c$ | CRC-32C of the 8 little-endian bytes of $s$ followed by $P$ | `u32` |
| $g(x)$ | the Castagnoli generator polynomial, `0x1EDC6F41` (reflected form `0x82F63B78`) | 33-bit polynomial over GF(2) |

A record is 16 header bytes and the payload, all integers little-endian:

```
offset  size  field
0       4     L        payload bytes
4       4     c        crc32c(seq bytes ++ payload)
8       8     s        sequence number
16      L     P        payload
```

A crash can stop a write at any byte, and a disk can return a flipped bit. The reader must tell a complete, correct record from anything else using only the bytes. Two checks do it: the length must fit in what remains of the file, and the checksum must match.

**CRC-32C.** Read the message as a polynomial over GF(2): each bit is a coefficient, addition is xor. The CRC is the remainder of dividing the message (shifted by 32 bits) by $g(x)$, with two conventions that make leading and trailing zero bytes count: the register starts at `0xFFFFFFFF` and the result is xored with `0xFFFFFFFF`. In code, one bit at a time, using the reflected polynomial because bits are processed least significant first:

```
c = 0xFFFFFFFF
for each byte b:  c ^= b
                  repeat 8 times: c = (c >> 1) ^ (0x82F63B78 if c & 1 else 0)
return c ^ 0xFFFFFFFF
```

Its standard check value is `crc32c("123456789") = 0xE3069283`. Any error burst of up to 32 bits is detected, and a random corruption slips through with probability $2^{-32}$. Go's `hash/crc32` with `crc32.MakeTable(crc32.Castagnoli)` computes the same function with a table (and CPU instructions); you may use it. The IEEE polynomial of zlib is a different function. Covering $s$ as well as $P$ means a valid record copied to the wrong position (a stale page, a duplicated write) fails the check against the sequence it claims.

### 2.2 Batches: one append, one record

The log stores **streams**: `run/<run_id>` for a workflow run's history, `queue/<name>` for a task queue. An `Append(stream, expected, evs...)` writes all its events as **one record**, so a crash leaves either all of them or none. The payload is the batch:

```
u16 len(stream), stream, u64 first_version, u32 count,
count x ( u16 len(type), type, i64 at_unix_nano, u32 len(data), data )
```

Versions are 1-based and dense per stream: a stream at version 7 that appends 3 events gets 8, 9, 10. The durable server puts a `tl.durable.v1.WalRecord` protobuf in each event's `data`, so the event bytes are those of `formats/wal.md`.

### 2.3 Segments

One ever-growing file is awkward to archive or truncate, so the log is a sequence of **segment** files named by the sequence number of their first record, as 20 decimal digits: `00000000000000000001.log`, then for example `00000000000000000004.log`. A new segment starts **before** a record that would take the current one past `SegmentBytes` (64 MiB by default), so no segment is ever larger than the limit unless a single record is. Creating a file is itself a change to the directory, so the writer `fsync`s the directory after creating a segment.

### 2.4 Recovery: torn tail or corruption

`Open` reads every segment in order and checks each record: the header fits, the length fits, the CRC matches, $s$ is the previous plus one, and the batch decodes with its first version right after the stream's last. The first record that fails decides everything:

| Where | What it means | What `Open` does |
|---|---|---|
| in the **last** segment | a crash in the middle of an append that was never acknowledged (a torn tail) | truncate the file at that record's offset and continue: the next append gets $s$ = last valid + 1 |
| in **any earlier** segment | bytes that were acknowledged are damaged | refuse to start with `ErrCorrupt`, naming the file and offset |

Truncating instead of skipping matters: if the garbage stays, the next record is written after it, and the following recovery stops at the garbage and drops the good record behind it.

### 2.5 Durability: write, fsync, then acknowledge

`write(2)` puts bytes in the kernel's page cache; a power cut can still lose them. `fsync(2)` returns only when the device has them. So `Append` is:

1. check the expected version and the quota;
2. roll to a new segment if needed;
3. write the record;
4. call `Sync` (an injectable function, `(*os.File).Sync` by default);
5. only then update memory and return the new version.

If 3 or 4 fails, `Append` truncates the segment back to its previous size before returning the error. A process killed by SIGKILL loses nothing the kernel already has, which is why the course tests may pass a plain `fsync(2)` through `Options.Sync`: on macOS Go's `(*os.File).Sync` is `F_FULLFSYNC`, about 4 ms per call, while `fsync(2)` takes microseconds. In production keep the default. Several concurrent appends may share one `fsync` (group commit); this module does one per append.

The failpoint `dur/log/after-write-before-fsync` (passed in as `Options.Failpoint`) sits between steps 3 and 4: a crash there leaves a whole record that was never acknowledged, which recovery may keep or drop, but never half of.

### 2.6 Optimistic concurrency and the quota

`Append(stream, expected, evs...)` fails with `ErrVersionConflict` unless the stream is at version `expected` (0 for a new stream); `Any` (−1) skips the check. Two server goroutines that both read version $v$ and both try to append as $v+1$: one wins, the other gets the conflict and rereads. No locks are held between the read and the append.

`MaxBytes` (`[durable].wal_max_bytes`) caps the total size of all segments. An append that would pass it fails with `ErrQuota` and writes nothing; the server maps it to gRPC `RESOURCE_EXHAUSTED`. Reads keep working, and raising the cap and restarting resumes writes (drill `ops.11`).

## 3. Worked example by hand

Append one event to a new log: stream `s`, type `T`, data `hi`, at time 0 ns, as the first record ever ($s = 1$, version 1).

The payload, field by field (32 bytes):

```
01 00                      len(stream) = 1
73                         "s"
01 00 00 00 00 00 00 00    first_version = 1
01 00 00 00                count = 1
01 00                      len(type) = 1
54                         "T"
00 00 00 00 00 00 00 00    at = 0 ns
02 00 00 00                len(data) = 2
68 69                      "hi"
```

The header: $L = 32$ = `20 00 00 00`; $s = 1$ = `01 00 00 00 00 00 00 00`; $c$ is CRC-32C over those 8 seq bytes followed by the 32 payload bytes, `0x0B721A5E`, stored little-endian as `5e 1a 72 0b`. The whole record is 48 bytes:

```
20 00 00 00  5e 1a 72 0b  01 00 00 00 00 00 00 00  01 00 73 01 00 ... 02 00 00 00 68 69
```

Now recovery. Suppose stream `s` received three appends of type `t` with data `one`, `two`, and `three`: records of 49, 49, and 51 bytes (16 + 15 + 18 + the data length). The process died while writing the third, after 20 of its 51 bytes. `Open` parses record 1 (seq 1, CRC ok), record 2 (seq 2, CRC ok), then at offset 98 finds a header saying $L = 35$ with only 4 payload bytes left: `ErrShort`. This is the last segment, so it truncates the file to 98 bytes, the stream is at version 2, and the next append writes seq 3 at offset 98. Had the same short record been in `00000000000000000001.log` with `00000000000000000004.log` after it, `Open` would have returned `ErrCorrupt` naming the first file.

These are `TestRecordHandExample` and `TestTornTailTruncatedAtEveryOffset`.

## 4. The interface

```go
package log // import "tinyllm/durable/log"

const HeaderSize = 16
var ErrShort, ErrChecksum, ErrBadBatch error            // ParseRecord, DecodeBatch

type Event struct {
	Stream  string
	Version int64     // 1-based, dense per stream; set by Append
	Type    string
	Data    []byte
	At      time.Time // set by Append from Options.Now
}

func CRC32C(b []byte) uint32
func AppendRecord(dst []byte, seq uint64, payload []byte) []byte
func ParseRecord(b []byte) (seq uint64, payload []byte, n int, err error) // never panics
func EncodeBatch(stream string, first int64, evs []Event) []byte
func DecodeBatch(p []byte) (stream string, evs []Event, err error)

const Any int64 = -1
const DefaultSegmentBytes int64 = 64 << 20
const FailpointAfterWrite = "dur/log/after-write-before-fsync"
var ErrVersionConflict, ErrQuota, ErrClosed, ErrCorrupt error

type Options struct {
	SegmentBytes int64                   // 0 = 64 MiB
	MaxBytes     int64                   // wal_max_bytes; 0 = no quota
	Now          func() time.Time        // nil = time.Now
	Sync         func(f *os.File) error  // nil = (*os.File).Sync
	Failpoint    func(name string) error // nil = none
}

func SegmentName(first uint64) string     // "%020d.log"
func Open(dir string, o Options) (*Log, error)
func (l *Log) Append(ctx context.Context, stream string, expected int64, evs ...Event) (int64, error)
func (l *Log) Read(ctx context.Context, stream string, from int64, limit int) ([]Event, error)
func (l *Log) Version(stream string) int64
func (l *Log) Streams(prefix string) []string // sorted
func (l *Log) LastSeq() uint64
func (l *Log) Size() int64
func (l *Log) Close() error
```

The log keeps every event in memory as well as on disk (fine for the course's histories; see Going further). Use only the standard library. `Read` returns events whose `Data` you must treat as read-only.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestRecordHandExample` | unit, golden | section 3's 32-byte payload and 48-byte record, against an independent CRC | the byte format `dur.10` replicates |
| `TestCRC32CCheckValue` | unit | `0xE3069283` for `"123456789"` | Castagnoli, not IEEE |
| `TestParseRecordRejectsDamage` | unit, boundary | every truncation is `ErrShort`, every flipped bit of seq or payload is `ErrChecksum` | recovery's only evidence |
| `TestDecoderNeverPanics` | property | 20,000 mutated payloads and records: an error or an exact round trip, never a panic | a crash leaves arbitrary bytes |
| `TestAppendReadHand` | unit | per-stream versions, `At`, `Read` from and limit, unknown stream, `Streams`, `LastSeq` | how `dur.02` reads a history |
| `TestAppendCopiesData` | unit | an event keeps its bytes after the caller reuses its buffer | the server's encode buffers |
| `TestVersionConflict` | unit, boundary | a wrong `expected` writes nothing; `Any` skips the check | two writers of one run |
| `TestConcurrentAppendersConflict` | property, fault | 32 goroutines, each success lands at expected+1, no event lost or doubled | the server appends from many rpcs |
| `TestReopenRecovers` | unit | identical reads, size, and seq after a restart; the next append continues | every restart |
| `TestSegmentsRollAndNameByFirstSeq` | unit, boundary | files `…1.log`, `…4.log`, `…7.log` at 170 bytes per segment, none over the limit | archiving, `dur.10` snapshots |
| `TestTornTailTruncatedAtEveryOffset` | property, fault | every cut inside the last record is truncated exactly there, twice-reopen safe | a crash mid-append |
| `TestGarbageTailTruncated` | boundary, fault | zeros, junk, a repeated record, an absurd length, half a header | the same, in the forms disks produce |
| `TestCorruptMiddleSegmentRefused` | fault, boundary | `ErrCorrupt` naming the damaged file, which is left untouched | acknowledged data is never silently dropped |
| `TestQuotaRejectsAndResumes` | fault, boundary | a record that fits exactly is accepted, the next is `ErrQuota` and changes nothing, a higher cap resumes | drill `ops.11` |
| `TestFailedAppendLeavesNoTrace` | fault | an injected failure after the write leaves no bytes behind | unacknowledged data must not reappear |
| `TestSyncBeforeAck` | fault | one `Sync` per record, after the write, before the return; none for a refused append | the durability promise |
| `TestKillLoopAckedSurvive` | fault | 30 SIGKILLs of a child appender; every printed version survives, densely | MS-durable's kill loop |
| `TestFailpointCrashAfterWrite` | fault | a crash between write and fsync keeps 5 acked versions and the 6th whole or not at all | the named failpoint of the contract |

`FuzzParseRecord` and `FuzzDecodeBatch` run the never-panics property under `go test -fuzz` in the nightly job.

Your own tests (rung R4) go in `go/durable/log/log_learner_test.go` as `package log_test`, and are written as properties rather than single cases: any truncation of the last record recovers to the records before it; every flipped bit is detected; for any sequence of appends, a reopen reads back exactly what was acknowledged; concurrent appenders never share a version; an append over the quota changes nothing. `ol check dur.01` runs them against the reference with one planted bug at a time (the mutants of section 5); they must catch 80% of them and every one marked required.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. a checksum over the payload only, the IEEE polynomial, or a different field width | records from another implementation (Raft, a backup tool) do not parse; a copied record passes | `TestRecordHandExample`, `TestCRC32CCheckValue` (mutants `s01`, `s02`, `s19`) |
| 2. trusting the length field without checking it against the bytes left after the header | a torn header with a huge length panics with a slice out of range | `TestParseRecordRejectsDamage` (mutant `s03`) |
| 3. recovery that skips a torn tail in memory but never truncates the file, or does not restore the seq counter | the next record lands after garbage (or reuses a seq) and the next recovery drops it | `TestTornTailTruncatedAtEveryOffset`, `TestReopenRecovers`, `TestKillLoopAckedSurvive` (mutants `s08`, `s11`) |
| 4. treating damage in an earlier segment as a torn tail | acknowledged history silently disappears | `TestCorruptMiddleSegmentRefused` (mutant `s12`) |
| 5. checking only `expected > current` | a stale writer overwrites a newer version: a lost update | `TestVersionConflict`, `TestConcurrentAppendersConflict` (mutant `s07`) |
| 6. off-by-one versions or reads | a history that starts at 0, skips its first event, or ignores the limit | `TestAppendReadHand` (mutants `s04`, `s05`, `s06`) |
| 7. naming a segment by the last seq written, or rolling after the limit is passed | recovery sees a segment whose name disagrees with its first record; segments grow past the limit | `TestSegmentsRollAndNameByFirstSeq` (mutants `s09`, `s10`) |
| 8. keeping the caller's slice, or encoding time in seconds | events change after `Append` returns; times change across a restart | `TestAppendCopiesData`, `TestReopenRecovers` (mutants `s13`, `s14`) |
| 9. no quota, or one that refuses a record that fits exactly | the disk fills and the node dies; or a write fails one record early | `TestQuotaRejectsAndResumes` (mutants `s15`, `s16`) |
| 10. returning an error from a failed write or fsync without cutting the bytes back | a never-acknowledged record is recovered after the next restart | `TestFailedAppendLeavesNoTrace` (mutant `s17`) |
| 11. acknowledging before `fsync` | a power cut loses events that callers were told are safe | `TestSyncBeforeAck` (mutant `s18`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.06` | goroutines, `context`, `os.File` |
| Back | `M06.3` | hashing ideas behind CRC-32C, defined here in section 2.1 |
| Forward | `dur.02` | each run's history is stream `run/<run_id>`, one `WalRecord` per event, appended with `expected` = the history's length |
| Forward | `dur.03` | each queue is stream `queue/<name>`; `Open` replays it to rebuild leases and dead letters |
| Forward | `dur.10` (optional) | Raft replicates exactly these framed records |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| segmented log | Temporal persistence (Cassandra, PostgreSQL, SQLite) | history shards so many servers write in parallel; history archival to blob storage | `temporalio/temporal`: `common/persistence` |
| every event in memory | an index of offsets per stream, reads from disk, segment compaction | histories larger than RAM; deleting finished runs | Kafka's log segments and indexes, `kafka/log` |
| one fsync per append | group commit | many concurrent appends share one fsync, so throughput rises with load | PostgreSQL `commit_delay`, RocksDB `WriteThread` |
| Continue-As-New (dur.06) bounds one run | history size limits and archival | a long-lived workflow never grows one history without bound | Temporal "Continue-As-New" docs |
