// Package log is the durable server's append-only segmented event log
// (dur.01; formats/wal.md). This file is the byte level: the framed record
// (len, CRC-32C, seq, payload) and the batch payload that carries one
// Append's events.
//
// Record, all integers little-endian (formats/wal.md):
//
//	offset  size  field
//	0       4     u32 len     payload bytes
//	4       4     u32 crc32c  CRC-32C (Castagnoli) of the 8 seq bytes, then the payload
//	8       8     u64 seq     1 for the first record ever, then +1, no gaps across segments
//	16      len   payload
//
// Payload (one Append = one record, so a batch is all or nothing):
//
//	u16 stream_len, stream, u64 first_version, u32 count,
//	count x (u16 type_len, type, i64 at_unix_nano, u32 data_len, data)
package log

import (
	"encoding/binary"
	"errors"
	"hash/crc32"
	"time"
)

// HeaderSize is the fixed part of a record before the payload.
const HeaderSize = 16

var (
	// ErrShort: fewer bytes than the header, or than the header says. At the
	// end of the last segment it is a torn tail; anywhere else, corruption.
	ErrShort = errors.New("log: short record")
	// ErrChecksum: the CRC-32C over seq and payload does not match.
	ErrChecksum = errors.New("log: checksum mismatch")
	// ErrBadBatch: the payload is not a well-formed batch.
	ErrBadBatch = errors.New("log: malformed batch payload")
)

var castagnoli = crc32.MakeTable(crc32.Castagnoli)

// Event is one entry of a stream. Version is 1-based and dense within its
// stream; the log sets Stream, Version, and At on Append.
type Event struct {
	Stream  string
	Version int64
	Type    string
	Data    []byte
	At      time.Time
}

// CRC32C is the Castagnoli CRC of b (check value: "123456789" -> 0xE3069283).
func CRC32C(b []byte) uint32 {
	return crc32.Checksum(b, castagnoli)
}

// recordCRC is the record checksum: CRC-32C over the 8 seq bytes, then the payload.
func recordCRC(seq uint64, payload []byte) uint32 {
	var s [8]byte
	binary.LittleEndian.PutUint64(s[:], seq)
	c := crc32.Update(0, castagnoli, s[:])
	return crc32.Update(c, castagnoli, payload)
}

// AppendRecord appends the framed record (header and payload) to dst.
func AppendRecord(dst []byte, seq uint64, payload []byte) []byte {
	var h [HeaderSize]byte
	binary.LittleEndian.PutUint32(h[0:4], uint32(len(payload)))
	binary.LittleEndian.PutUint32(h[4:8], recordCRC(seq, payload))
	binary.LittleEndian.PutUint64(h[8:16], seq)
	dst = append(dst, h[:]...)
	return append(dst, payload...)
}

// ParseRecord reads the record at the start of b. n is its total size
// (HeaderSize + len); payload aliases b. It never allocates and never panics
// on any input (the fuzz target checks that).
func ParseRecord(b []byte) (seq uint64, payload []byte, n int, err error) {
	if len(b) < HeaderSize {
		return 0, nil, 0, ErrShort
	}
	ln := uint64(binary.LittleEndian.Uint32(b[0:4]))
	if ln > uint64(len(b)-HeaderSize) {
		return 0, nil, 0, ErrShort
	}
	crc := binary.LittleEndian.Uint32(b[4:8])
	seq = binary.LittleEndian.Uint64(b[8:16])
	payload = b[HeaderSize : HeaderSize+int(ln)]
	if recordCRC(seq, payload) != crc {
		return 0, nil, 0, ErrChecksum
	}
	return seq, payload, HeaderSize + int(ln), nil
}

// EncodeBatch is the payload of one Append: the stream, the version of the
// first event, and each event's type, time, and data.
func EncodeBatch(stream string, first int64, evs []Event) []byte {
	n := 2 + len(stream) + 8 + 4
	for _, e := range evs {
		n += 2 + len(e.Type) + 8 + 4 + len(e.Data)
	}
	b := make([]byte, 0, n)
	b = binary.LittleEndian.AppendUint16(b, uint16(len(stream)))
	b = append(b, stream...)
	b = binary.LittleEndian.AppendUint64(b, uint64(first))
	b = binary.LittleEndian.AppendUint32(b, uint32(len(evs)))
	for _, e := range evs {
		b = binary.LittleEndian.AppendUint16(b, uint16(len(e.Type)))
		b = append(b, e.Type...)
		b = binary.LittleEndian.AppendUint64(b, uint64(e.At.UnixNano()))
		b = binary.LittleEndian.AppendUint32(b, uint32(len(e.Data)))
		b = append(b, e.Data...)
	}
	return b
}

// DecodeBatch is the inverse of EncodeBatch. It sets Stream, Version (first,
// first+1, ...), Type, At, and a copy of Data on every event, and rejects a
// payload with a zero count, a first version below 1, or bytes left over.
func DecodeBatch(p []byte) (stream string, evs []Event, err error) {
	r := reader{b: p}
	stream = string(r.bytes(int(r.u16())))
	first := int64(r.u64())
	count := r.u32()
	if r.bad || first < 1 || count == 0 || uint64(count) > uint64(len(p)) {
		return "", nil, ErrBadBatch
	}
	evs = make([]Event, 0, count)
	for i := uint32(0); i < count; i++ {
		typ := string(r.bytes(int(r.u16())))
		at := int64(r.u64())
		data := append([]byte(nil), r.bytes(int(r.u32()))...)
		if r.bad {
			return "", nil, ErrBadBatch
		}
		evs = append(evs, Event{Stream: stream, Version: first + int64(i), Type: typ, Data: data, At: time.Unix(0, at)})
	}
	if len(r.b) != 0 {
		return "", nil, ErrBadBatch
	}
	return stream, evs, nil
}

// reader consumes little-endian fields; bad latches on the first short read.
type reader struct {
	b   []byte
	bad bool
}

func (r *reader) bytes(n int) []byte {
	if r.bad || n < 0 || n > len(r.b) {
		r.bad = true
		return nil
	}
	out := r.b[:n]
	r.b = r.b[n:]
	return out
}

func (r *reader) u16() uint16 {
	if b := r.bytes(2); b != nil {
		return binary.LittleEndian.Uint16(b)
	}
	return 0
}

func (r *reader) u32() uint32 {
	if b := r.bytes(4); b != nil {
		return binary.LittleEndian.Uint32(b)
	}
	return 0
}

func (r *reader) u64() uint64 {
	if b := r.bytes(8); b != nil {
		return binary.LittleEndian.Uint64(b)
	}
	return 0
}
