// Package raft replicates the durable server's log across three replicas
// (dur.10, optional). It is the consensus core of Ongaro and Ousterhout,
// "In Search of an Understandable Consensus Algorithm" (2014), figure 2:
// leader election, log replication, and the commit rule.
//
// A Node is a deterministic state machine. It does no I/O and starts no
// goroutine: time advances only when the caller calls Tick, messages arrive
// only through Step, and everything the node wants to send is collected for
// ReadMessages. The caller moves messages between nodes (gRPC tl.raft.v1 in
// the server, an in-process network in the course tests), and applies the
// entries ReadCommitted returns, in index order. Durable state (the current
// term, the vote, the log) goes through Storage before any message that
// depends on it is queued, which is what makes a crash and restart safe.
//
// Each entry's Data is one framed WAL record (formats/wal.md), so the server
// replicates exactly the bytes its single-node log writes. A new leader
// appends one entry with empty Data (a no-op) so that entries of earlier terms
// can commit; appliers skip it.
package raft

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"slices"
)

// Role is a node's place in its current term.
type Role int

const (
	Follower Role = iota
	Candidate
	Leader
)

func (r Role) String() string {
	switch r {
	case Follower:
		return "follower"
	case Candidate:
		return "candidate"
	case Leader:
		return "leader"
	}
	return fmt.Sprintf("Role(%d)", int(r))
}

// Entry is one log entry. Index is 1-based; Term is the term of the leader
// that created it. Empty Data is the leader's no-op.
type Entry struct {
	Index uint64
	Term  uint64
	Data  []byte
}

// MsgType names the four messages of figure 2 (tl.raft.v1 RequestVote and
// AppendEntries, request and response).
type MsgType int

const (
	MsgVote     MsgType = iota + 1 // RequestVote
	MsgVoteResp                    // RequestVote response
	MsgApp                         // AppendEntries; a heartbeat when Entries is empty
	MsgAppResp                     // AppendEntries response
)

func (t MsgType) String() string {
	switch t {
	case MsgVote:
		return "MsgVote"
	case MsgVoteResp:
		return "MsgVoteResp"
	case MsgApp:
		return "MsgApp"
	case MsgAppResp:
		return "MsgAppResp"
	}
	return fmt.Sprintf("MsgType(%d)", int(t))
}

// Message is one RPC or reply between two nodes. Fields that a type does not
// use are zero.
type Message struct {
	Type MsgType
	From uint64
	To   uint64
	Term uint64 // the sender's current term

	// MsgVote: the candidate's last log entry.
	LastLogIndex uint64
	LastLogTerm  uint64

	// MsgVoteResp
	Granted bool

	// MsgApp: the entry just before Entries, the entries, and the leader's
	// commit index.
	PrevLogIndex uint64
	PrevLogTerm  uint64
	Entries      []Entry
	Commit       uint64

	// MsgAppResp. On success MatchIndex is the last index the follower now
	// holds in agreement with the leader. On a log mismatch ConflictIndex is
	// the follower's hint for the next PrevLogIndex + 1 to try.
	Success       bool
	MatchIndex    uint64
	ConflictIndex uint64
}

// HardState is what a node must remember across a crash besides its log.
type HardState struct {
	Term uint64 // the latest term this node has seen
	Vote uint64 // the candidate it voted for in Term; 0 = none
}

// Storage is the node's stable storage. A Node calls it synchronously and
// queues no message that depends on a write until the write returned nil.
type Storage interface {
	// Load returns what was last saved; a fresh store returns the zero
	// HardState and no entries. Entries are contiguous from index 1.
	Load() (HardState, []Entry, error)
	SaveHardState(HardState) error
	// Append persists entries (contiguous, non-empty). When entries[0].Index
	// is at or below the last stored index, every stored entry from
	// entries[0].Index on is replaced.
	Append(entries []Entry) error
}

// Config of one node.
type Config struct {
	ID    uint64   // this node's id, >= 1
	Peers []uint64 // every member's id, this node included
	// ElectionTicks is E: a follower or candidate that hears from no leader
	// and grants no vote for its election timeout, a value in [E, 2E) drawn
	// again at every reset, starts an election.
	ElectionTicks int
	// HeartbeatTicks: a leader sends AppendEntries to every peer this often.
	// It must be below ElectionTicks.
	HeartbeatTicks int
	// Timeout, when set, replaces the random draw (tests fix it).
	Timeout func() int
	// Seed seeds the election-timeout draw (a PCG stream per node id).
	Seed uint64
	// MaxEntries caps the entries in one MsgApp; 0 = 64.
	MaxEntries int
}

var (
	// ErrNotLeader is wrapped by *NotLeaderError.
	ErrNotLeader = errors.New("raft: not the leader")
	// ErrEmptyProposal: empty Data is reserved for the leader's no-op.
	ErrEmptyProposal = errors.New("raft: empty proposal")
)

// NotLeaderError is returned by Propose on a node that is not the leader.
// Leader is the leader this node knows in its current term, 0 when it knows
// none (during an election). The server turns it into UNAVAILABLE with the
// trailer tl-raft-leader.
type NotLeaderError struct{ Leader uint64 }

func (e *NotLeaderError) Error() string {
	if e.Leader == 0 {
		return "raft: not the leader (no leader known)"
	}
	return fmt.Sprintf("raft: not the leader (leader is %d)", e.Leader)
}

func (e *NotLeaderError) Unwrap() error { return ErrNotLeader }

// Status is a snapshot of a node's volatile and durable state.
type Status struct {
	ID        uint64
	Term      uint64
	Vote      uint64
	Role      Role
	Leader    uint64 // 0 = unknown in this term
	Commit    uint64
	LastIndex uint64
	LastTerm  uint64
}

// Node is one Raft member. It is not safe for concurrent use: the caller
// serializes Tick, Step, Propose, and the reads.
type Node struct {
	cfg Config
	st  Storage
	rng *rand.Rand

	term uint64
	vote uint64
	log  []Entry // log[i].Index == i+1

	commit  uint64
	applied uint64 // entries up to here were returned by ReadCommitted

	role    Role
	leader  uint64
	elapsed int // ticks since the timer was last reset
	timeout int // the current election timeout, in ticks

	votes map[uint64]bool   // candidate: who granted
	next  map[uint64]uint64 // leader: next index to send each peer
	match map[uint64]uint64 // leader: highest index known replicated on each peer

	out []Message
	err error // sticky: a storage failure stops the node
}

// New loads the node's state from st and starts it as a follower with no
// known leader and commit index 0 (the commit index is volatile, figure 2:
// after a restart committed entries are returned again from index 1, and the
// applier must tolerate that, by replaying into a fresh state or by skipping
// what it already holds).
func New(cfg Config, st Storage) (*Node, error) {
	if cfg.ID == 0 {
		return nil, errors.New("raft: id must be >= 1")
	}
	if !slices.Contains(cfg.Peers, cfg.ID) {
		return nil, fmt.Errorf("raft: id %d is not in peers %v", cfg.ID, cfg.Peers)
	}
	seen := map[uint64]bool{}
	for _, p := range cfg.Peers {
		if p == 0 || seen[p] {
			return nil, fmt.Errorf("raft: peers %v must be distinct ids >= 1", cfg.Peers)
		}
		seen[p] = true
	}
	if cfg.HeartbeatTicks <= 0 || cfg.ElectionTicks <= cfg.HeartbeatTicks {
		return nil, fmt.Errorf("raft: want 0 < HeartbeatTicks (%d) < ElectionTicks (%d)", cfg.HeartbeatTicks, cfg.ElectionTicks)
	}
	if cfg.MaxEntries <= 0 {
		cfg.MaxEntries = 64
	}
	hs, ents, err := st.Load()
	if err != nil {
		return nil, fmt.Errorf("raft: load: %w", err)
	}
	for i, e := range ents {
		if e.Index != uint64(i+1) {
			return nil, fmt.Errorf("raft: stored entry %d has index %d", i+1, e.Index)
		}
	}
	n := &Node{
		cfg:  cfg,
		st:   st,
		rng:  rand.New(rand.NewPCG(cfg.Seed, cfg.ID)),
		term: hs.Term,
		vote: hs.Vote,
		log:  slices.Clone(ents),
		role: Follower,
	}
	n.resetTimer()
	return n, nil
}

// Tick advances the node's clock by one tick: a leader sends heartbeats
// every HeartbeatTicks, anyone else starts an election when its timeout
// passes.
func (n *Node) Tick() error {
	if n.err != nil {
		return n.err
	}
	n.elapsed++
	if n.role == Leader {
		if n.elapsed >= n.cfg.HeartbeatTicks {
			n.elapsed = 0
			n.broadcastAppend()
		}
		return nil
	}
	if n.elapsed >= n.timeout {
		n.campaign()
	}
	return n.err
}

// Step handles one message addressed to this node.
func (n *Node) Step(m Message) error {
	if n.err != nil {
		return n.err
	}
	if m.To != n.cfg.ID {
		return fmt.Errorf("raft: node %d got a message for %d", n.cfg.ID, m.To)
	}
	if m.Term > n.term {
		// A newer term: whatever we were, we follow it now. Only an
		// AppendEntries names the leader of that term.
		lead := uint64(0)
		if m.Type == MsgApp {
			lead = m.From
		}
		n.becomeFollower(m.Term, lead)
		if n.err != nil {
			return n.err
		}
	}
	if m.Term < n.term {
		// Stale sender: answer requests with our term so it steps down;
		// drop stale responses.
		switch m.Type {
		case MsgVote:
			n.send(Message{Type: MsgVoteResp, To: m.From, Granted: false})
		case MsgApp:
			n.send(Message{Type: MsgAppResp, To: m.From, Success: false})
		}
		return nil
	}
	switch m.Type {
	case MsgVote:
		n.handleVote(m)
	case MsgVoteResp:
		n.handleVoteResp(m)
	case MsgApp:
		n.handleAppend(m)
	case MsgAppResp:
		n.handleAppendResp(m)
	default:
		return fmt.Errorf("raft: unknown message type %v", m.Type)
	}
	return n.err
}

// Propose appends data to the leader's log and starts replicating it. It
// returns the entry's index and term: the proposal is committed when an
// entry with that index AND that term comes out of ReadCommitted. If another
// entry commits at that index, the proposal was lost (a leader change) and
// the client may retry. A follower or candidate returns *NotLeaderError.
func (n *Node) Propose(data []byte) (index, term uint64, err error) {
	if n.err != nil {
		return 0, 0, n.err
	}
	if len(data) == 0 {
		return 0, 0, ErrEmptyProposal
	}
	if n.role != Leader {
		return 0, 0, &NotLeaderError{Leader: n.leader}
	}
	e := Entry{Index: n.lastIndex() + 1, Term: n.term, Data: slices.Clone(data)}
	if !n.appendLocal(e) {
		return 0, 0, n.err
	}
	n.broadcastAppend()
	n.maybeCommit()
	return e.Index, e.Term, nil
}

// ReadMessages returns the messages queued since the last call, in the order
// they were produced, and empties the queue.
func (n *Node) ReadMessages() []Message {
	out := n.out
	n.out = nil
	return out
}

// ReadCommitted returns the entries committed since the last call, in index
// order (no-ops included), and marks them applied.
func (n *Node) ReadCommitted() []Entry {
	if n.commit <= n.applied {
		return nil
	}
	out := cloneEntries(n.log[n.applied:n.commit])
	n.applied = n.commit
	return out
}

// Status reports the node's state.
func (n *Node) Status() Status {
	return Status{
		ID: n.cfg.ID, Term: n.term, Vote: n.vote, Role: n.role, Leader: n.leader,
		Commit: n.commit, LastIndex: n.lastIndex(), LastTerm: n.lastTerm(),
	}
}

// -- elections --------------------------------------------------------------------

// campaign starts an election: a new term, a vote for itself, saved before
// any RequestVote leaves.
func (n *Node) campaign() {
	n.role = Candidate
	n.term++
	n.vote = n.cfg.ID
	n.leader = 0
	if !n.saveHardState() {
		return
	}
	n.resetTimer()
	n.votes = map[uint64]bool{n.cfg.ID: true}
	if n.quorum(len(n.votes)) {
		n.becomeLeader()
		return
	}
	for _, p := range n.cfg.Peers {
		if p != n.cfg.ID {
			n.send(Message{Type: MsgVote, To: p, LastLogIndex: n.lastIndex(), LastLogTerm: n.lastTerm()})
		}
	}
}

// handleVote grants at most one vote per term, and only to a candidate whose
// log is at least as up to date as ours (section 5.4.1): a later last term
// wins; with equal last terms, the longer log wins.
func (n *Node) handleVote(m Message) {
	canVote := n.vote == 0 || n.vote == m.From
	upToDate := m.LastLogTerm > n.lastTerm() ||
		(m.LastLogTerm == n.lastTerm() && m.LastLogIndex >= n.lastIndex())
	grant := canVote && upToDate
	if grant {
		n.vote = m.From
		if !n.saveHardState() {
			return
		}
		n.resetTimer() // granting a vote defers our own candidacy
	}
	n.send(Message{Type: MsgVoteResp, To: m.From, Granted: grant})
}

func (n *Node) handleVoteResp(m Message) {
	if n.role != Candidate || !m.Granted {
		return
	}
	n.votes[m.From] = true
	if n.quorum(len(n.votes)) {
		n.becomeLeader()
	}
}

// becomeFollower moves to term (saving it with no vote when it is new) and
// follows lead (0 = not known yet).
func (n *Node) becomeFollower(term, lead uint64) {
	if term > n.term {
		n.term = term
		n.vote = 0
		if !n.saveHardState() {
			return
		}
	}
	n.role = Follower
	n.leader = lead
	n.votes, n.next, n.match = nil, nil, nil
	n.resetTimer()
}

// becomeLeader initializes nextIndex and matchIndex, appends the term's
// no-op, and sends it to everyone.
func (n *Node) becomeLeader() {
	n.role = Leader
	n.leader = n.cfg.ID
	n.elapsed = 0
	n.votes = nil
	n.next = map[uint64]uint64{}
	n.match = map[uint64]uint64{}
	for _, p := range n.cfg.Peers {
		n.next[p] = n.lastIndex() + 1
		n.match[p] = 0
	}
	if !n.appendLocal(Entry{Index: n.lastIndex() + 1, Term: n.term}) {
		return
	}
	n.broadcastAppend()
	n.maybeCommit()
}

// -- replication ------------------------------------------------------------------

// handleAppend is the follower side of AppendEntries (figure 2, receiver
// implementation, steps 2 to 5).
func (n *Node) handleAppend(m Message) {
	// Same term: the sender is this term's one leader.
	if n.role != Follower || n.leader != m.From {
		n.becomeFollower(m.Term, m.From)
		if n.err != nil {
			return
		}
	}
	n.resetTimer()
	reply := Message{Type: MsgAppResp, To: m.From}
	if m.PrevLogIndex > n.lastIndex() {
		reply.ConflictIndex = n.lastIndex() + 1
		n.send(reply)
		return
	}
	if t := n.termAt(m.PrevLogIndex); t != m.PrevLogTerm {
		// Skip the whole conflicting term in one round trip.
		first := m.PrevLogIndex
		for first > 1 && n.termAt(first-1) == t {
			first--
		}
		reply.ConflictIndex = first
		n.send(reply)
		return
	}
	// Find the first entry we lack or hold with another term. Entries we
	// already hold with the same term stay: this message may be an old,
	// shorter copy, and truncating would drop entries the leader counted.
	for i, e := range m.Entries {
		if e.Index <= n.lastIndex() && n.termAt(e.Index) == e.Term {
			continue
		}
		rest := cloneEntries(m.Entries[i:])
		if err := n.st.Append(rest); err != nil {
			n.fail(err)
			return
		}
		n.log = append(n.log[:e.Index-1], rest...)
		break
	}
	lastNew := m.PrevLogIndex + uint64(len(m.Entries))
	if c := min(m.Commit, lastNew); c > n.commit {
		n.commit = c
	}
	reply.Success = true
	reply.MatchIndex = lastNew
	n.send(reply)
}

// handleAppendResp is the leader side: advance matchIndex on success, back
// nextIndex off on a mismatch, then send what the follower still lacks.
func (n *Node) handleAppendResp(m Message) {
	if n.role != Leader {
		return
	}
	p := m.From
	if m.Success {
		if m.MatchIndex > n.match[p] {
			n.match[p] = m.MatchIndex
		}
		if n.next[p] < n.match[p]+1 {
			n.next[p] = n.match[p] + 1
		}
		n.maybeCommit()
		if n.next[p] <= n.lastIndex() {
			n.sendAppend(p)
		}
		return
	}
	next := n.next[p] - 1
	if m.ConflictIndex >= 1 && m.ConflictIndex < next {
		next = m.ConflictIndex
	}
	n.next[p] = max(next, n.match[p]+1, 1)
	n.sendAppend(p)
}

// maybeCommit advances the commit index to the highest index N of the
// leader's current term that a majority holds (section 5.4.2: an entry of
// an earlier term is never committed by counting replicas; it commits when
// an entry of the current term after it does).
func (n *Node) maybeCommit() {
	for idx := n.lastIndex(); idx > n.commit; idx-- {
		if n.termAt(idx) != n.term {
			break // earlier terms: only indirectly, below
		}
		count := 0
		for _, p := range n.cfg.Peers {
			if p == n.cfg.ID || n.match[p] >= idx {
				count++
			}
		}
		if n.quorum(count) {
			n.commit = idx
			return
		}
	}
}

// broadcastAppend sends AppendEntries to every peer: new entries where the
// peer lags, an empty heartbeat (with the commit index) where it does not.
func (n *Node) broadcastAppend() {
	for _, p := range n.cfg.Peers {
		if p != n.cfg.ID {
			n.sendAppend(p)
		}
	}
}

func (n *Node) sendAppend(to uint64) {
	next := n.next[to]
	prev := next - 1
	end := min(n.lastIndex(), prev+uint64(n.cfg.MaxEntries))
	var ents []Entry
	if next <= end {
		ents = cloneEntries(n.log[prev:end])
	}
	n.send(Message{
		Type: MsgApp, To: to, PrevLogIndex: prev, PrevLogTerm: n.termAt(prev),
		Entries: ents, Commit: n.commit,
	})
}

// -- helpers ----------------------------------------------------------------------

func (n *Node) quorum(count int) bool {
	return count > len(n.cfg.Peers)/2
}

// appendLocal appends one entry the leader created, saved before use.
func (n *Node) appendLocal(e Entry) bool {
	if err := n.st.Append([]Entry{e}); err != nil {
		n.fail(err)
		return false
	}
	n.log = append(n.log, e)
	return true
}

func (n *Node) saveHardState() bool {
	if err := n.st.SaveHardState(HardState{Term: n.term, Vote: n.vote}); err != nil {
		n.fail(err)
		return false
	}
	return true
}

// fail stops the node: after a storage error it may have promised something
// it cannot remember, so it sends nothing more, including what was queued.
func (n *Node) fail(err error) {
	n.err = fmt.Errorf("raft: node %d storage: %w", n.cfg.ID, err)
	n.out = nil
}

func (n *Node) resetTimer() {
	n.elapsed = 0
	if n.cfg.Timeout != nil {
		n.timeout = n.cfg.Timeout()
	} else {
		e := n.cfg.ElectionTicks
		n.timeout = e + n.rng.IntN(e)
	}
}

func (n *Node) send(m Message) {
	m.From = n.cfg.ID
	m.Term = n.term
	n.out = append(n.out, m)
}

func (n *Node) lastIndex() uint64 { return uint64(len(n.log)) }

func (n *Node) lastTerm() uint64 { return n.termAt(n.lastIndex()) }

// termAt is the term of the entry at idx; 0 for idx 0 (the empty prefix).
func (n *Node) termAt(idx uint64) uint64 {
	if idx == 0 || idx > uint64(len(n.log)) {
		return 0
	}
	return n.log[idx-1].Term
}

func cloneEntries(es []Entry) []Entry {
	out := make([]Entry, len(es))
	for i, e := range es {
		out[i] = Entry{Index: e.Index, Term: e.Term, Data: slices.Clone(e.Data)}
	}
	return out
}
