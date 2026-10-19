package log

// This file is the segmented log: segment files named by their first seq,
// recovery (refuse corruption, truncate a torn tail), optimistic version
// checks per stream, the wal_max_bytes quota, and the fsync before an
// append is acknowledged.

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	// Any skips the expected-version check of Append.
	Any int64 = -1
	// DefaultSegmentBytes is the size past which a new segment starts.
	DefaultSegmentBytes int64 = 64 << 20
	// FailpointAfterWrite fires after the record is written, before fsync:
	// a crash there is exactly the torn-tail case of recovery.
	FailpointAfterWrite = "dur/log/after-write-before-fsync"
)

var (
	ErrVersionConflict = errors.New("log: version conflict")
	ErrQuota           = errors.New("log: wal_max_bytes reached") // the server maps it to RESOURCE_EXHAUSTED
	ErrClosed          = errors.New("log: closed")
	ErrCorrupt         = errors.New("log: corrupt segment") // Open refuses to start
)

// Options configures a Log. The zero value is the production default.
type Options struct {
	SegmentBytes int64                   // 0 = DefaultSegmentBytes
	MaxBytes     int64                   // [durable].wal_max_bytes; 0 = no quota
	Now          func() time.Time        // stamps Event.At; nil = time.Now
	Sync         func(f *os.File) error  // nil = (*os.File).Sync (F_FULLFSYNC on macOS)
	Failpoint    func(name string) error // nil = none; called at FailpointAfterWrite
}

// SegmentName is the file name of the segment whose first record has seq first.
func SegmentName(first uint64) string { return fmt.Sprintf("%020d.log", first) }

type segment struct {
	first uint64
	path  string
	size  int64
}

// Log is safe for concurrent use; appends are serialized.
type Log struct {
	mu      sync.Mutex
	dir     string
	o       Options
	segs    []segment
	f       *os.File // the last segment, open for append
	seq     uint64   // seq of the last record written
	size    int64    // bytes over every segment
	streams map[string][]Event
	closed  bool
}

// Open recovers the log in dir (created if missing). In every segment but the
// last, a short record, a bad checksum, a seq gap, or a malformed batch is
// ErrCorrupt naming the file and offset; in the last segment the first such
// record is a torn tail and the file is truncated there.
func Open(dir string, o Options) (*Log, error) {
	if o.SegmentBytes <= 0 {
		o.SegmentBytes = DefaultSegmentBytes
	}
	if o.Now == nil {
		o.Now = time.Now
	}
	if o.Sync == nil {
		o.Sync = (*os.File).Sync
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	names, err := segmentFiles(dir)
	if err != nil {
		return nil, err
	}
	l := &Log{dir: dir, o: o, streams: map[string][]Event{}}
	for i, s := range names {
		last := i == len(names)-1
		data, err := os.ReadFile(s.path)
		if err != nil {
			return nil, err
		}
		if len(data) > 0 && s.first != l.seq+1 {
			return nil, fmt.Errorf("%w: %s starts at seq %d, want %d", ErrCorrupt, filepath.Base(s.path), s.first, l.seq+1)
		}
		off := 0
		for off < len(data) {
			seq, payload, n, perr := ParseRecord(data[off:])
			var stream string
			var evs []Event
			if perr == nil && seq != l.seq+1 {
				perr = fmt.Errorf("seq %d after %d", seq, l.seq)
			}
			if perr == nil {
				stream, evs, perr = DecodeBatch(payload)
			}
			if perr == nil && evs[0].Version != int64(len(l.streams[stream]))+1 {
				perr = fmt.Errorf("stream %q version %d after %d", stream, evs[0].Version, len(l.streams[stream]))
			}
			if perr != nil {
				if !last {
					return nil, fmt.Errorf("%w: %s offset %d: %v", ErrCorrupt, filepath.Base(s.path), off, perr)
				}
				if err := os.Truncate(s.path, int64(off)); err != nil {
					return nil, err
				}
				data = data[:off]
				break
			}
			l.streams[stream] = append(l.streams[stream], evs...)
			l.seq = seq
			off += n
		}
		s.size = int64(len(data))
		l.size += s.size
		l.segs = append(l.segs, s)
	}
	if len(l.segs) > 0 {
		last := l.segs[len(l.segs)-1]
		f, err := os.OpenFile(last.path, os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return nil, err
		}
		l.f = f
	}
	return l, nil
}

// segmentFiles lists dir's segment files in seq order.
func segmentFiles(dir string) ([]segment, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out []segment
	for _, e := range ents {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".log") || len(name) != 24 {
			continue
		}
		first, err := strconv.ParseUint(strings.TrimSuffix(name, ".log"), 10, 64)
		if err != nil {
			continue
		}
		out = append(out, segment{first: first, path: filepath.Join(dir, name)})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].first < out[j].first })
	return out, nil
}

// Append writes evs to stream as one record and returns the stream's new
// version. expected is the version the caller last saw (0 for a new stream),
// or Any; a mismatch is ErrVersionConflict and writes nothing. The record is
// fsynced before Append returns; an append past MaxBytes is ErrQuota; any
// failure leaves the log exactly as it was.
func (l *Log) Append(ctx context.Context, stream string, expected int64, evs ...Event) (int64, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return 0, ErrClosed
	}
	cur := int64(len(l.streams[stream]))
	if expected != Any && expected != cur {
		return cur, fmt.Errorf("%w: stream %q is at version %d, expected %d", ErrVersionConflict, stream, cur, expected)
	}
	if len(evs) == 0 {
		return cur, nil
	}
	now := l.o.Now()
	batch := make([]Event, len(evs))
	for i, e := range evs {
		batch[i] = Event{Stream: stream, Version: cur + 1 + int64(i), Type: e.Type, Data: append([]byte(nil), e.Data...), At: now}
	}
	rec := AppendRecord(nil, l.seq+1, EncodeBatch(stream, cur+1, batch))
	if l.o.MaxBytes > 0 && l.size+int64(len(rec)) > l.o.MaxBytes {
		return cur, fmt.Errorf("%w: %d + %d bytes over %d", ErrQuota, l.size, len(rec), l.o.MaxBytes)
	}
	if err := l.roll(int64(len(rec))); err != nil {
		return cur, err
	}
	seg := &l.segs[len(l.segs)-1]
	if err := l.write(seg, rec); err != nil {
		return cur, err
	}
	l.seq++
	seg.size += int64(len(rec))
	l.size += int64(len(rec))
	l.streams[stream] = append(l.streams[stream], batch...)
	return cur + int64(len(evs)), nil
}

// write appends rec to the open segment, then the failpoint, then fsync. On
// any error it truncates the segment back, so the failed record can never be
// recovered as if it had been acknowledged.
func (l *Log) write(seg *segment, rec []byte) error {
	undo := func(err error) error {
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
	if err := l.o.Sync(l.f); err != nil {
		return undo(err)
	}
	return nil
}

// roll starts a new segment, named by the seq of the record about to be
// written, when there is none yet or when n more bytes would take the current
// one past SegmentBytes (a segment is never left empty).
func (l *Log) roll(n int64) error {
	if len(l.segs) > 0 {
		cur := l.segs[len(l.segs)-1]
		if cur.size == 0 || cur.size+n <= l.o.SegmentBytes {
			return nil
		}
	}
	first := l.seq + 1
	path := filepath.Join(l.dir, SegmentName(first))
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	if err := syncDir(l.dir); err != nil {
		f.Close()
		os.Remove(path)
		return err
	}
	if l.f != nil {
		l.f.Close()
	}
	l.f = f
	l.segs = append(l.segs, segment{first: first, path: path})
	return nil
}

// syncDir makes a new segment's directory entry durable.
func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}

// Read returns up to limit events of stream (limit <= 0: all) starting at
// version from (from < 1 reads from 1). An unknown stream is empty.
func (l *Log) Read(ctx context.Context, stream string, from int64, limit int) ([]Event, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil, ErrClosed
	}
	evs := l.streams[stream]
	if from < 1 {
		from = 1
	}
	if from > int64(len(evs)) {
		return nil, nil
	}
	out := evs[from-1:]
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return append([]Event(nil), out...), nil
}

// Version is the number of events in stream (0 when it does not exist).
func (l *Log) Version(stream string) int64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	return int64(len(l.streams[stream]))
}

// Streams lists the streams whose name starts with prefix, sorted.
func (l *Log) Streams(prefix string) []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	var out []string
	for s := range l.streams {
		if strings.HasPrefix(s, prefix) {
			out = append(out, s)
		}
	}
	sort.Strings(out)
	return out
}

// LastSeq is the seq of the last record (0 for an empty log).
func (l *Log) LastSeq() uint64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.seq
}

// Size is the total size of every segment in bytes (what MaxBytes caps).
func (l *Log) Size() int64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.size
}

// Close closes the open segment; later calls fail with ErrClosed.
func (l *Log) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil
	}
	l.closed = true
	if l.f != nil {
		return l.f.Close()
	}
	return nil
}
