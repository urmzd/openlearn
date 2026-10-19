# Append-only segmented event log

## Overview

| | |
|---|---|
| **Module** | `dur.01` · build · Go · Pass 8 · 4 to 6 h |
| **Solution** | [`go/durable/log/record.go`](../../../../course/solutions/go/durable/log/record.go): `CRC32C`, `AppendRecord`, `ParseRecord`, `EncodeBatch`, `DecodeBatch`; [`go/durable/log/log.go`](../../../../course/solutions/go/durable/log/log.go): `Open`, `Append`, `Read`, `Version`, `Streams`, `LastSeq`, `Size`, `Close`; model property tests in [`log_learner_test.go`](../../../../course/solutions/learner-tests/dur.01/go/durable/log/log_learner_test.go) |
| **Contract** | the bytes on disk are [`formats/wal.md`](../../../../course/contracts/formats/wal.md) (records, segments, recovery, quota); the Go API is section 4 of this chapter |
| **Needs** | reading: [`lang.06` Go](../../programming-languages/02-language-and-tool-primers/06-go.md), [`lang.10` Protocol Buffers and gRPC](../../programming-languages/02-language-and-tool-primers/10-protocol-buffers-and-grpc.md), [`M06.3` hashing](../../mathematics-for-computing/06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) |
| **Used by** | `dur.02` keeps every workflow run's history in it · `dur.03` keeps every queue transition in it · the base of the durable engine, which survives a SIGKILL loop end to end |
| **Optional depth** | Kleppmann, *Designing Data-Intensive Applications*, ch. 3 (logs) and 7; Pillai et al., *All File Systems Are Not Created Equal* (OSDI 2014); the Castagnoli paper, *Optimization of Cyclic Redundancy-Check Codes with 24 and 32 Parity Bits* (1993) |

## Key Takeaways

- A log record is framed as length, CRC-32C, sequence number, payload; the CRC covers the sequence number too, so a record copied to the wrong place is caught.
- Recovery never guesses: damage in the **last** segment is a torn tail from a crash mid-append and is truncated; damage anywhere **before** it is lost acknowledged data and stops the server.
- An append is acknowledged only after `write` **and** `fsync`; a failure in between cuts the bytes back off, or a record nobody acknowledged comes back after a restart.
- Streams give optimistic concurrency: `Append(stream, expected, ...)` succeeds only if the stream is still at `expected`, so two writers can never both write version $v+1$.
- SIGKILL the writer at 30 random instants and every acknowledged version is still there.

## How to read this chapter

Sections 1 to 3 build the log from first principles and work one record and one recovery by hand. Section 4 fixes the interface. Section 5 walks through the solution in the order it is best read (`record.go` first, then `Open`, `Append`, and segment rolling in `log.go`), explains why each choice is the right one, and shows the model property tests. Try to predict each step of the solution before reading it.

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

Its standard check value is `crc32c("123456789") = 0xE3069283`. Any error burst of up to 32 bits is detected, and a random corruption slips through with probability $2^{-32}$. Go's `hash/crc32` with `crc32.MakeTable(crc32.Castagnoli)` computes the same function with a table (and CPU instructions); the solution uses it. The IEEE polynomial of zlib is a different function. Covering $s$ as well as $P$ means a valid record copied to the wrong position (a stale page, a duplicated write) fails the check against the sequence it claims.

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

If 3 or 4 fails, `Append` truncates the segment back to its previous size before returning the error. A process killed by SIGKILL loses nothing the kernel already has, which is why tests may pass a plain `fsync(2)` through `Options.Sync`: on macOS Go's `(*os.File).Sync` is `F_FULLFSYNC`, about 4 ms per call, while `fsync(2)` takes microseconds. In production keep the default. Several concurrent appends may share one `fsync` (group commit); this module does one per append.

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

The log keeps every event in memory as well as on disk (fine for the course's histories; see Going further). The solution uses only the standard library. `Read` returns events whose `Data` callers must treat as read-only.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example record | unit, golden | section 3's 32-byte payload and 48-byte record, against an independent CRC | the byte format `dur.10` replicates |
| CRC-32C check value | unit | `0xE3069283` for `"123456789"` | Castagnoli, not IEEE |
| parse rejects damage | unit, boundary | every truncation is `ErrShort`, every flipped bit of seq or payload is `ErrChecksum` | recovery's only evidence |
| decoder never panics | property | 20,000 mutated payloads and records: an error or an exact round trip, never a panic | a crash leaves arbitrary bytes |
| append and read by hand | unit | per-stream versions, `At`, `Read` from and limit, unknown stream, `Streams`, `LastSeq` | how `dur.02` reads a history |
| append copies data | unit | an event keeps its bytes after the caller reuses its buffer | the server's encode buffers |
| version conflict | unit, boundary | a wrong `expected` writes nothing; `Any` skips the check | two writers of one run |
| concurrent appenders | property, fault | 32 goroutines, each success lands at expected+1, no event lost or doubled | the server appends from many rpcs |
| reopen recovers | unit | identical reads, size, and seq after a restart; the next append continues | every restart |
| segments roll, named by first seq | unit, boundary | files `…1.log`, `…4.log`, `…7.log` at 170 bytes per segment, none over the limit | archiving, `dur.10` snapshots |
| torn tail at every offset | property, fault | every cut inside the last record is truncated exactly there, twice-reopen safe | a crash mid-append |
| garbage tail truncated | boundary, fault | zeros, junk, a repeated record, an absurd length, half a header | the same, in the forms disks produce |
| corrupt middle segment refused | fault, boundary | `ErrCorrupt` naming the damaged file, which is left untouched | acknowledged data is never silently dropped |
| quota rejects and resumes | fault, boundary | a record that fits exactly is accepted, the next is `ErrQuota` and changes nothing, a higher cap resumes | drill `ops.11` |
| failed append leaves no trace | fault | an injected failure after the write leaves no bytes behind | unacknowledged data must not reappear |
| sync before ack | fault | one `Sync` per record, after the write, before the return; none for a refused append | the durability promise |
| kill loop: acked versions survive | fault | 30 SIGKILLs of a child appender; every printed version survives, densely | the durable engine's kill loop |
| crash at the failpoint after write | fault | a crash between write and fsync keeps 5 acked versions and the 6th whole or not at all | the named failpoint of the contract |

The never-panics property is also stated as two fuzz targets, `FuzzParseRecord` and `FuzzDecodeBatch`, for `go test -fuzz`.

## 5. The solution, step by step

### From first principles

1. **The log's two promises are "nothing acknowledged is lost" and "nothing half-written is believed".** The first forces `fsync` before returning; the second forces every record to be self-validating from its own bytes, because after a crash the bytes are the only evidence.
2. **Self-validating means length plus checksum.** The length says where the record ends; the CRC says whether those bytes are the ones written. Including the sequence number under the CRC means a correct record in the wrong place also fails.
3. **One `Append` must be atomic.** A workflow task that completes and schedules two activities writes three events; recovering one of them would be a history no code ever produced. So one `Append` is one record, and a batch inside it carries every event.
4. **Recovery has to classify damage, and position is the only classifier.** Only the record being written when the process died can be torn, and it is always the last one. Damage anywhere earlier was acknowledged, so it is data loss: refuse to start rather than run on a silently shorter history.
5. **A torn tail must be removed, not skipped.** If it stays, the next record is appended after it, and the next recovery stops at the garbage and drops the good record. So `Open` truncates the file.
6. **Concurrency control belongs at the append.** Holding a lock from a server's read until its append would serialise every RPC. An expected version checked inside the append's critical section gives the same safety with no lock held across the round trip.
7. **The quota and every failure must leave the log unchanged.** Anything that refuses or fails must happen before the write or be undone by truncation, so a caller that sees an error can retry without a phantom record.

### The code, annotated

**Framing a record** ([`record.go`](../../../../course/solutions/go/durable/log/record.go)). The checksum covers the seq bytes then the payload, computed incrementally so no concatenated buffer is built.

```go
var castagnoli = crc32.MakeTable(crc32.Castagnoli) // not crc32.IEEETable

func recordCRC(seq uint64, payload []byte) uint32 {
	var s [8]byte
	binary.LittleEndian.PutUint64(s[:], seq)
	c := crc32.Update(0, castagnoli, s[:])    // seq first: a record moved to another seq fails
	return crc32.Update(c, castagnoli, payload)
}

func AppendRecord(dst []byte, seq uint64, payload []byte) []byte {
	var h [HeaderSize]byte
	binary.LittleEndian.PutUint32(h[0:4], uint32(len(payload)))
	binary.LittleEndian.PutUint32(h[4:8], recordCRC(seq, payload))
	binary.LittleEndian.PutUint64(h[8:16], seq)
	dst = append(dst, h[:]...)
	return append(dst, payload...)
}
```

**Parsing without trusting the header.** Every check happens before slicing, so arbitrary bytes produce an error, never a panic.

```go
func ParseRecord(b []byte) (seq uint64, payload []byte, n int, err error) {
	if len(b) < HeaderSize {
		return 0, nil, 0, ErrShort // a torn header
	}
	ln := uint64(binary.LittleEndian.Uint32(b[0:4]))
	// Compare in uint64 against what is actually left: a torn length field can
	// claim 4 GiB, and slicing with it would panic.
	if ln > uint64(len(b)-HeaderSize) {
		return 0, nil, 0, ErrShort
	}
	crc := binary.LittleEndian.Uint32(b[4:8])
	seq = binary.LittleEndian.Uint64(b[8:16])
	payload = b[HeaderSize : HeaderSize+int(ln)] // aliases b: no allocation
	if recordCRC(seq, payload) != crc {
		return 0, nil, 0, ErrChecksum
	}
	return seq, payload, HeaderSize + int(ln), nil
}
```

`DecodeBatch` uses a small `reader` whose `bad` flag latches on the first short read, so the decoder can read every field in sequence and check once. It rejects a zero count, a first version below 1, a count larger than the payload could hold, and leftover bytes: a well-formed batch is exactly its fields. It copies each event's `Data` so the events do not alias the segment buffer.

**Recovery in `Open`** ([`log.go`](../../../../course/solutions/go/durable/log/log.go)). Each segment is read whole; each record must pass four checks in order.

```go
for i, s := range names {
	last := i == len(names)-1
	data, err := os.ReadFile(s.path)
	// ...
	if len(data) > 0 && s.first != l.seq+1 { // the file name must agree with its first record
		return nil, fmt.Errorf("%w: %s starts at seq %d, want %d", ErrCorrupt, filepath.Base(s.path), s.first, l.seq+1)
	}
	off := 0
	for off < len(data) {
		seq, payload, n, perr := ParseRecord(data[off:])     // 1. frame and CRC
		var stream string
		var evs []Event
		if perr == nil && seq != l.seq+1 {                    // 2. dense seq, no gaps or repeats
			perr = fmt.Errorf("seq %d after %d", seq, l.seq)
		}
		if perr == nil {
			stream, evs, perr = DecodeBatch(payload)          // 3. well-formed batch
		}
		if perr == nil && evs[0].Version != int64(len(l.streams[stream]))+1 { // 4. dense versions
			perr = fmt.Errorf("stream %q version %d after %d", stream, evs[0].Version, len(l.streams[stream]))
		}
		if perr != nil {
			if !last {
				return nil, fmt.Errorf("%w: %s offset %d: %v", ErrCorrupt, filepath.Base(s.path), off, perr)
			}
			// Torn tail: cut the file so the next append lands on clean bytes.
			if err := os.Truncate(s.path, int64(off)); err != nil {
				return nil, err
			}
			data = data[:off]
			break
		}
		l.streams[stream] = append(l.streams[stream], evs...)
		l.seq = seq // the seq counter is rebuilt from what survived
		off += n
	}
	s.size = int64(len(data))
	l.size += s.size
	l.segs = append(l.segs, s)
}
```

The seq check is what catches a repeated record (a duplicated write) in the tail: its CRC is valid, but its seq is not the next one. Recovery is idempotent: a second `Open` of a truncated log finds nothing to truncate.

**`Append`: check, encode, quota, roll, write, then publish.**

```go
l.mu.Lock()
defer l.mu.Unlock()
// ...
cur := int64(len(l.streams[stream]))
if expected != Any && expected != cur { // equality, not ">": a stale writer must fail too
	return cur, fmt.Errorf("%w: stream %q is at version %d, expected %d", ErrVersionConflict, stream, cur, expected)
}
// ...
now := l.o.Now()
batch := make([]Event, len(evs))
for i, e := range evs {
	// Copy Data: the caller may reuse its buffer after Append returns.
	batch[i] = Event{Stream: stream, Version: cur + 1 + int64(i), Type: e.Type, Data: append([]byte(nil), e.Data...), At: now}
}
rec := AppendRecord(nil, l.seq+1, EncodeBatch(stream, cur+1, batch))
if l.o.MaxBytes > 0 && l.size+int64(len(rec)) > l.o.MaxBytes { // ">": a record that fits exactly is accepted
	return cur, fmt.Errorf("%w: %d + %d bytes over %d", ErrQuota, l.size, len(rec), l.o.MaxBytes)
}
if err := l.roll(int64(len(rec))); err != nil {
	return cur, err
}
seg := &l.segs[len(l.segs)-1]
if err := l.write(seg, rec); err != nil {
	return cur, err
}
// Only now, after fsync, does memory change: readers never see an unsynced event.
l.seq++
seg.size += int64(len(rec))
l.size += int64(len(rec))
l.streams[stream] = append(l.streams[stream], batch...)
return cur + int64(len(evs)), nil
```

The record is fully encoded before the quota check, so the check uses the exact size on disk. Times are stored as `UnixNano` in the batch, so `At` survives a restart unchanged.

**`write`: write, failpoint, fsync, or undo.**

```go
func (l *Log) write(seg *segment, rec []byte) error {
	undo := func(err error) error {
		// seg.size has not been advanced yet, so it is the size before this record.
		if terr := l.f.Truncate(seg.size); terr != nil {
			return errors.Join(err, terr)
		}
		return err
	}
	if _, err := l.f.Write(rec); err != nil {
		return undo(err)
	}
	if l.o.Failpoint != nil {
		if err := l.o.Failpoint(FailpointAfterWrite); err != nil {
			return undo(err)
		}
	}
	if err := l.o.Sync(l.f); err != nil { // the durability point
		return undo(err)
	}
	return nil
}
```

**`roll`: name by the first seq, roll before the limit.**

```go
func (l *Log) roll(n int64) error {
	if len(l.segs) > 0 {
		cur := l.segs[len(l.segs)-1]
		// Keep writing here if the segment is empty (never leave one empty) or
		// if this record still fits.
		if cur.size == 0 || cur.size+n <= l.o.SegmentBytes {
			return nil
		}
	}
	first := l.seq + 1 // the record about to be written names the file
	path := filepath.Join(l.dir, SegmentName(first))
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND|os.O_CREATE|os.O_EXCL, 0o644)
	// ...
	if err := syncDir(l.dir); err != nil { // the new directory entry must be durable too
		f.Close()
		os.Remove(path)
		return err
	}
	// ... close the old file, switch to f, append the segment
}
```

`O_EXCL` refuses to overwrite an existing segment, which would be a bug elsewhere. `Read` returns a copy of the slice of events (`append([]Event(nil), out...)`) so callers cannot change the log's in-memory history by appending to it.

Full solution: [`record.go`](../../../../course/solutions/go/durable/log/record.go), [`log.go`](../../../../course/solutions/go/durable/log/log.go).

### The model property tests

This module's solution includes a test suite written as properties, in [`log_learner_test.go`](../../../../course/solutions/learner-tests/dur.01/go/durable/log/log_learner_test.go). The torn-tail property is the template: write three records, then for every cut point inside the last one, check that recovery lands on exactly the first two, truncates the file, continues at the right seq, and is stable across a second reopen.

```go
for cut := good; cut < full; cut++ {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, dlog.SegmentName(1)), data[:cut], 0o644)
	l2, err := dlog.Open(dir, dlog.Options{Sync: fastSync})
	if err != nil {
		t.Fatalf("cut at %d of %d: Open failed: %v (a torn tail is not corruption)", cut, full, err)
	}
	if l2.Version("s") != 2 || l2.LastSeq() != 2 || l2.Size() != good { /* ... */ }
	if st, _ := os.Stat(filepath.Join(dir, dlog.SegmentName(1))); st.Size() != good { /* not truncated */ }
	if v, err := l2.Append(ctx, "s", 2, ev("t", "again")); err != nil || v != 3 || l2.LastSeq() != 3 { /* ... */ }
	l2.Close()
	// a second reopen must agree: recovery is idempotent
}
```

Testing every offset rather than one is what makes it a property: a bug that only appears when the cut falls inside the header, or exactly at the payload boundary, cannot hide. The suite's kill loop runs the appender as a child process, SIGKILLs it at random instants, and checks every version it printed as acknowledged is still there, densely.

### Why this solution

- **Cost.** An append is O(record size) to encode and write plus one `fsync`, which dominates (microseconds to milliseconds depending on the device). `Open` is O(total log size): it reads every byte once. Reads are O(1) to locate and O(k) to copy k events, because every event is also in memory.
- **CRC-32C over a cryptographic hash.** The threat is accidental damage, not an adversary; CRC-32C detects all bursts up to 32 bits, has hardware support, and is what other WAL tools speak.
- **Truncate-on-tail, refuse-on-middle, over "skip bad records".** Skipping hides data loss and makes recovery depend on what follows the damage. The chosen rule is decidable from position alone.
- **Expected-version over locks.** No lock is held across a client round trip, and conflicts are explicit errors the server turns into retries.
- **Accepted tradeoffs.** All events live in memory (fine for course-sized histories; production keeps an offset index and reads from disk), and one `fsync` per append caps throughput at the device's fsync rate (group commit is the fix, in Going further).

### Explain it back

1. Why does the CRC cover the sequence number and not just the payload?

   <details><summary>Answer</summary>

   A stale page or a duplicated write can put a perfectly valid record at the wrong position. Its payload CRC would still match. With the seq under the CRC, and recovery requiring seq to be the previous plus one, the misplaced record fails either the checksum or the density check.

   </details>

2. Damage in the last segment is truncated, damage in an earlier one refuses to start. What justifies treating them differently?

   <details><summary>Answer</summary>

   Only the record being appended when the process died can be incomplete, and appends only go to the last segment. That record was never acknowledged, so dropping it loses nothing anyone was promised. Every record before it was fsynced and acknowledged, so damage there is real data loss, which must be surfaced, not hidden.

   </details>

3. `write` fails after the bytes reached the file but before `fsync`. What would go wrong without `undo`?

   <details><summary>Answer</summary>

   The caller gets an error and believes nothing was written, but the record is whole on disk (or in the page cache). After a restart, recovery would find a valid record and resurrect an event that was reported as failed, possibly duplicating it when the caller retries.

   </details>

4. Why does `Append` check `expected != cur` rather than `expected > cur`?

   <details><summary>Answer</summary>

   A writer that read version 5 when the stream is now at 7 has `expected < cur`. With only a "greater" check it would be allowed to append at 8 based on a stale view, a lost update. Equality is the only condition under which the writer's view is current.

   </details>

5. Why is the quota check `l.size + len(rec) > MaxBytes` computed after the record is encoded?

   <details><summary>Answer</summary>

   The quota caps bytes on disk, and only the encoded record has its exact size (header, stream name, types, data). Checking with `>` means a record that lands exactly on the cap is accepted and the next is refused, with nothing written.

   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. a checksum over the payload only, the IEEE polynomial, or a different field width | records from another implementation (Raft, a backup tool) do not parse; a copied record passes | `recordCRC` uses the Castagnoli table over seq then payload, with the fixed 16-byte header |
| 2. trusting the length field without checking it against the bytes left after the header | a torn header with a huge length panics with a slice out of range | `ParseRecord` compares the length with `len(b)-HeaderSize` before slicing |
| 3. recovery that skips a torn tail in memory but never truncates the file, or does not restore the seq counter | the next record lands after garbage (or reuses a seq) and the next recovery drops it | `Open` calls `os.Truncate` at the bad offset and rebuilds `l.seq` from surviving records |
| 4. treating damage in an earlier segment as a torn tail | acknowledged history silently disappears | `Open` returns `ErrCorrupt` unless the damage is in the `last` segment |
| 5. checking only `expected > current` | a stale writer overwrites a newer version: a lost update | `Append` requires `expected == cur` under `l.mu` |
| 6. off-by-one versions or reads | a history that starts at 0, skips its first event, or ignores the limit | versions start at `cur + 1`; `Read` slices `evs[from-1:]` and applies `limit` |
| 7. naming a segment by the last seq written, or rolling after the limit is passed | recovery sees a segment whose name disagrees with its first record; segments grow past the limit | `roll` names the file `l.seq + 1` and rolls when `cur.size+n` would exceed the limit |
| 8. keeping the caller's slice, or encoding time in seconds | events change after `Append` returns; times change across a restart | `Append` copies `Data`; `EncodeBatch` stores `UnixNano` |
| 9. no quota, or one that refuses a record that fits exactly | the disk fills and the node dies; or a write fails one record early | the check is `l.size+len(rec) > MaxBytes`, before anything is written |
| 10. returning an error from a failed write or fsync without cutting the bytes back | a never-acknowledged record is recovered after the next restart | `write`'s `undo` truncates back to `seg.size` |
| 11. acknowledging before `fsync` | a power cut loses events that callers were told are safe | `write` calls `Sync` before `Append` updates memory or returns |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.06` | goroutines, `context`, `os.File` |
| Back | `M06.3` | hashing ideas behind CRC-32C, defined here in section 2.1 |
| Forward | `dur.02` | each run's history is stream `run/<run_id>`, one `WalRecord` per event, appended with `expected` = the history's length |
| Forward | `dur.03` | each queue is stream `queue/<name>`; `Open` replays it to rebuild leases and dead letters |
| Forward | `dur.10` (optional) | Raft replicates exactly these framed records |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| segmented log | Temporal persistence (Cassandra, PostgreSQL, SQLite) | history shards so many servers write in parallel; history archival to blob storage | `temporalio/temporal`: `common/persistence` |
| every event in memory | an index of offsets per stream, reads from disk, segment compaction | histories larger than RAM; deleting finished runs | Kafka's log segments and indexes, `kafka/log` |
| one fsync per append | group commit | many concurrent appends share one fsync, so throughput rises with load | PostgreSQL `commit_delay`, RocksDB `WriteThread` |
| Continue-As-New (dur.06) bounds one run | history size limits and archival | a long-lived workflow never grows one history without bound | Temporal "Continue-As-New" docs |
