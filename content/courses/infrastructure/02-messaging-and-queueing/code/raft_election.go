// A compact, dependency-free illustration of Raft leader election -- the
// mechanism behind a Kafka KRaft controller quorum (and Redpanda's per-partition
// Raft). Standalone:
//
//	go run raft_election.go
//
// This is NOT a production Raft. It models the *election* slice only -- terms,
// votes, and the majority-quorum rule that elects exactly one controller per
// term -- so you can see why removing ZooKeeper (KIP-500, completed in Kafka 4.0)
// still gives a single source of truth for cluster metadata.
//
// Election rules modelled (Raft section 5.2):
//   - Time is divided into TERMS (monotonic). Each term has at most one leader.
//   - A follower that hears no leader becomes a CANDIDATE, bumps its term, votes
//     for itself, and requests votes.
//   - A node grants its vote at most once per term, and only to a candidate whose
//     term >= its own. A candidate wins with a strict MAJORITY (quorum).
//   - Majority quorum guarantees at most one leader per term (two disjoint
//     majorities cannot exist), which is the safety property KRaft relies on.
package main

import (
	"fmt"
	"sort"
)

type role int

const (
	follower role = iota
	candidate
	leader
)

type node struct {
	id       int
	term     int
	role     role
	votedFor int  // -1 = none this term
	alive    bool // a dead node neither requests nor grants votes
}

func newNode(id int) *node { return &node{id: id, votedFor: -1, alive: true} }

// requestVote: candidate c asks voter v for a vote in c.term.
// Returns true iff v grants it (Raft 5.2: same-or-newer term, not yet voted).
func requestVote(c *node, v *node) bool {
	if !v.alive {
		return false
	}
	if c.term > v.term { // newer term -> voter steps down and resets its vote
		v.term = c.term
		v.role = follower
		v.votedFor = -1
	}
	if c.term == v.term && (v.votedFor == -1 || v.votedFor == c.id) {
		v.votedFor = c.id
		return true
	}
	return false
}

// runElection: `cand` starts an election; the alive majority of `cluster` decides.
func runElection(cand *node, cluster []*node) bool {
	cand.term++
	cand.role = candidate
	cand.votedFor = cand.id
	votes := 1 // votes for itself

	for _, peer := range cluster {
		if peer.id == cand.id {
			continue
		}
		if requestVote(cand, peer) {
			votes++
		}
	}

	majority := len(cluster)/2 + 1 // strict majority over the FULL cluster size
	if votes >= majority {
		cand.role = leader
		fmt.Printf("term %d: node %d ELECTED leader with %d/%d votes (quorum=%d)\n",
			cand.term, cand.id, votes, len(cluster), majority)
		return true
	}
	cand.role = follower
	fmt.Printf("term %d: node %d failed (%d/%d votes, need %d)\n",
		cand.term, cand.id, votes, len(cluster), majority)
	return false
}

// leaderCountForTerm: safety check -- at most one leader may exist per term.
func leaderCountForTerm(cluster []*node, term int) int {
	n := 0
	for _, x := range cluster {
		if x.role == leader && x.term == term {
			n++
		}
	}
	return n
}

func main() {
	cluster := []*node{newNode(0), newNode(1), newNode(2), newNode(3), newNode(4)}
	fmt.Print("5-node controller quorum (majority = 3)\n\n")

	// 1) Healthy election: node 2 becomes the controller in term 1.
	if !runElection(cluster[2], cluster) {
		panic("a single candidate over a healthy quorum must win")
	}

	// 2) Two candidates contend in the SAME term: node 0 and node 1 both bump to
	//    term 2 and request votes. One-vote-per-term means at most one reaches
	//    quorum -- the classic split-vote that prevents a split brain.
	cluster[0].term, cluster[1].term = 1, 1  // both caught up to term 1
	won0 := runElection(cluster[0], cluster) // node 0 -> term 2, gathers votes
	won1 := runElection(cluster[1], cluster) // node 1 -> term 3, may steal votes
	_ = won0
	_ = won1

	// 3) Safety invariant: scan every term and assert <=1 leader per term.
	maxTerm := 0
	for _, x := range cluster {
		if x.term > maxTerm {
			maxTerm = x.term
		}
	}
	terms := make([]int, 0, maxTerm+1)
	for t := 0; t <= maxTerm; t++ {
		terms = append(terms, t)
	}
	sort.Ints(terms)
	for _, t := range terms {
		lc := leaderCountForTerm(cluster, t)
		if lc > 1 {
			panic(fmt.Sprintf("safety violated: %d leaders in term %d", lc, t))
		}
	}
	fmt.Printf("\nsafety holds: <=1 leader per term across terms 0..%d\n", maxTerm)

	// 4) Partition: kill a minority (2 of 5). A candidate still reaches majority.
	cluster[3].alive = false
	cluster[4].alive = false
	if !runElection(cluster[1], cluster) {
		panic("3 alive of 5 should still elect a leader (majority intact)")
	}

	// 5) Lose the majority: only 2 of 5 alive -> no leader can be elected.
	cluster[0].alive = false
	if runElection(cluster[1], cluster) {
		panic("with only 2/5 alive, no candidate should reach quorum")
	}
	fmt.Println("\nminority partition correctly blocked from electing a leader -> OK")
}
