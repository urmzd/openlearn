// kvstore serves the toy tl.kv.v1.KvTransferService over gRPC.
//
//	kvstore --port <n>     listen on 127.0.0.1:<n> (0: any free port)
//
// Prints `listening on 127.0.0.1:<port>` first; SIGTERM stops gracefully
// (in-flight calls finish) and exits 0.
package main

import (
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"syscall"

	"google.golang.org/grpc"

	"lang10/kvstore"
	kvv1 "openlearn.urmzd.com/tl/contracts/gen/tl/kv/v1"
)

func main() {
	// SOLUTION-BEGIN lang.10
	port := flag.Int("port", 50052, "port on 127.0.0.1 (0: any free port)")
	flag.Parse()
	lis, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", *port))
	if err != nil {
		fmt.Fprintln(os.Stderr, "kvstore:", err)
		os.Exit(1)
	}
	// Every message is capped at 4 MiB (DESIGN 2.7).
	srv := grpc.NewServer(grpc.MaxRecvMsgSize(4<<20), grpc.MaxSendMsgSize(4<<20))
	kvv1.RegisterKvTransferServiceServer(srv, kvstore.New())
	// Install the handler before announcing readiness: a SIGTERM that lands
	// between the two would otherwise kill the process outright.
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGTERM, os.Interrupt)
	fmt.Printf("listening on %s\n", lis.Addr())
	go func() {
		<-sig
		srv.GracefulStop()
	}()
	if err := srv.Serve(lis); err != nil {
		fmt.Fprintln(os.Stderr, "kvstore:", err)
		os.Exit(1)
	}
	// SOLUTION-END
}
