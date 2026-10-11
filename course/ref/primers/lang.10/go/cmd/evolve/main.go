// evolve reads one KvChunk in the protobuf wire format, as hex on stdin,
// decodes it with the generated type, and prints one JSON line: the known
// fields (defaults included) and the message encoded again, as hex.
//
//	echo 0a01681802 | evolve
//	{"handle_id":"h","block_hash":0,"block_index":2,"n_blocks_total":0,"kv_format":0,"payload_len":0,"crc32c":0,"reencoded":"0a01681802"}
//
// A field this version of the schema does not know survives the round trip.
package main

import (
	"bufio"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"google.golang.org/protobuf/proto"

	kvv1 "openlearn.urmzd.com/tl/contracts/gen/tl/kv/v1"
)

// Fields is what evolve prints.
type Fields struct {
	HandleID     string `json:"handle_id"`
	BlockHash    uint64 `json:"block_hash"`
	BlockIndex   uint32 `json:"block_index"`
	NBlocksTotal uint32 `json:"n_blocks_total"`
	KvFormat     uint32 `json:"kv_format"`
	PayloadLen   int    `json:"payload_len"`
	Crc32c       uint32 `json:"crc32c"`
	Reencoded    string `json:"reencoded"`
}

// Evolve decodes wire bytes into a KvChunk and encodes it again.
func Evolve(wire []byte) (Fields, error) {
	// SOLUTION-BEGIN lang.10
	var c kvv1.KvChunk
	if err := proto.Unmarshal(wire, &c); err != nil {
		return Fields{}, err
	}
	out, err := proto.Marshal(&c)
	if err != nil {
		return Fields{}, err
	}
	return Fields{
		HandleID:     c.GetHandleId(),
		BlockHash:    c.GetBlockHash(),
		BlockIndex:   c.GetBlockIndex(),
		NBlocksTotal: c.GetNBlocksTotal(),
		KvFormat:     c.GetKvFormat(),
		PayloadLen:   len(c.GetPayload()),
		Crc32c:       c.GetCrc32C(),
		Reencoded:    hex.EncodeToString(out),
	}, nil
	// SOLUTION-END
}

func main() {
	// SOLUTION-BEGIN lang.10
	line, _ := bufio.NewReader(os.Stdin).ReadString('\n')
	wire, err := hex.DecodeString(strings.TrimSpace(line))
	if err != nil {
		fmt.Fprintln(os.Stderr, "evolve: stdin is not hex:", err)
		os.Exit(2)
	}
	f, err := Evolve(wire)
	if err != nil {
		fmt.Fprintln(os.Stderr, "evolve:", err)
		os.Exit(1)
	}
	b, _ := json.Marshal(f)
	fmt.Println(string(b))
	// SOLUTION-END
}
