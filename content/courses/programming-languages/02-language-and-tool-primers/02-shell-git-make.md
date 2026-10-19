# Shell, git, make, processes: exit codes, signals, environment, pipes

## Overview

| | |
|---|---|
| **Module** | `lang.02` · practice · C and shell · Pass 0 · 1 to 2 h reading |
| **Solution** | [`Makefile`](../../../../course/solutions/primers/lang.02/Makefile) for the given two-file C program ([`main.c`](../../../../course/solutions/primers/lang.02/main.c), [`greet.c`](../../../../course/solutions/primers/lang.02/greet.c), [`greet.h`](../../../../course/solutions/primers/lang.02/greet.h)), and [`worker.sh`](../../../../course/solutions/primers/lang.02/worker.sh), a script that holds a lock file, traps SIGTERM and SIGINT (`stop`), cleans up, and exits 128 + the signal number |
| **Contract** | none: a primer exercise, not part of the system |
| **Needs** | a terminal with `bash`, `make`, `cc`, and `git` to run the examples; [`lang.01`](01-python-and-numpy.md) comes first on the path |
| **Used by** | no code call site (a primer). It is the concept prerequisite of `craft.01` (the CI gate is processes and exit codes), `M03.1` and `rt.01` (the system's C `Makefile`), `L10.0` (the engine shuts down on SIGTERM), and `dep.00` (Kubernetes stops a pod with SIGTERM, then SIGKILL) |
| **Optional depth** | *The Linux Command Line*, Shotts, ch. 6, 10, 24 to 27 (free at linuxcommand.org); GNU Make manual, ch. 2 and 4 (free); `man 7 signal`; *Pro Git*, ch. 1 to 3 (free at git-scm.com) |

## Key Takeaways

- **Every process ends with an exit status** from 0 to 255; 0 means success. CI, `make`, and Kubernetes all decide by that one number.
- **128 + n means "ended by signal n".** A trap that cleans up and exits 143 after SIGTERM keeps that information; a script without a trap is simply killed.
- **Bash runs a trap only between commands.** Sleep in the background and `wait`, or a 30 s `sleep` delays shutdown by up to 30 s.
- **SIGKILL cannot be trapped**, so anything cleaned up on exit can be left behind; the next start must detect stale state.
- **make rebuilds a target when a prerequisite is newer**, and it only knows the prerequisites you list: headers included.

## How to read this chapter

Section 2 builds the process model: exit status, signals, environment, pipes, then make and git. Section 3 works out by hand which targets a change rebuilds and what a worker's life looks like from the shell. Section 4 states the requirements for the Makefile and the worker and the cases a correct solution must handle. Section 5 walks through both solution files line by line and explains why each line is there. Try to write each rule and each line of the trap before reading it.

---

## 1. Why now

Next, craft.01 sets up the gate every later change to the system passes through: CI jobs that run commands and decide pass or fail by their exit status. From then on, the system's C code in Pass 1 is built by a `Makefile`, and its Rust engine and Go gateway run as processes that Kubernetes stops by sending SIGTERM and, 30 seconds later, SIGKILL. A server that ignores SIGTERM loses in-flight requests; a build that does not track headers ships objects compiled against an old struct layout; a script that ignores a failing command in a pipe reports green on a broken build. This primer teaches the process model those all share, on two small artifacts that run in seconds.

## 2. Principles

### 2.1 Processes and exit status

A **process** is a running program. It has a numeric id (its **PID**), a parent process, an environment (2.3), and three open files numbered 0, 1, 2 (2.4). A shell runs a command by creating a child process and waiting for it. When the child ends, it reports an **exit status**, an integer from 0 to 255 (only the low 8 bits of `exit(n)` survive, so `exit 256` reports 0). The shell exposes it and a few other facts as special parameters:

| Symbol | Meaning | Type / shape |
|---|---|---|
| `$?` | exit status of the last command | integer 0 to 255 |
| `$$` | PID of the current shell (in a script: the script's own PID) | integer |
| `$!` | PID of the last command started in the background with `&` | integer |
| `$1`, `$2`, ... | the script's arguments | strings |
| $n$ | a signal number: 2 for SIGINT, 9 for SIGKILL, 15 for SIGTERM | integer |
| $128 + n$ | the status a shell reports for a child ended by signal $n$ | integer |

Conventions every tool in this course follows: **0** success; **1** a general failure; **2** wrong usage (bad arguments); **126** found but not executable; **127** command not found; **$128 + n$** ended by signal $n$, so 130 after Ctrl-C, 137 after SIGKILL (what an out-of-memory kill looks like), 143 after SIGTERM.

Status drives control flow: `a && b` runs `b` only if `a` exited 0; `a || b` only if it did not. Three settings make scripts fail fast instead of carrying on after an error: `set -e` (exit when a command fails), `set -u` (an unset variable is an error), and `set -o pipefail` (2.4).

### 2.2 Signals

A **signal** is a small asynchronous message the kernel delivers to a process: "stop now", "you were interrupted". `kill -TERM 4242` sends SIGTERM to PID 4242; Ctrl-C in a terminal sends SIGINT. Each signal has a default action, and for these three it is "terminate":

| Signal | Number | Sent by | Can a process catch it? |
|---|---|---|---|
| SIGINT | 2 | Ctrl-C | yes |
| SIGTERM | 15 | `kill` with no option, Kubernetes, systemd, CI cancellation | yes |
| SIGKILL | 9 | `kill -9`, the out-of-memory killer, Kubernetes after the grace period | **no**: the process ends, nothing of it runs |

`kill -0 PID` sends nothing; it only tells you whether the process exists (status 0) or not.

In bash, `trap 'commands' TERM` replaces the default action with your commands. The rule that surprises everyone: **bash runs a trap only after the command it is currently running finishes.** While a foreground `sleep 30` runs, a SIGTERM waits. The builtin `wait`, however, returns as soon as a trapped signal arrives. So a script that must stop promptly starts its long waits in the background (`sleep 30 &`) and then `wait`s for them. Once the trap runs, the background child is still alive; the trap must kill it (`jobs -p` lists the PIDs of the shell's background children), or it lives on as an **orphan**.

### 2.3 Environment

The **environment** is a list of `NAME=value` strings each process carries. A child gets a **copy** of its parent's environment when it starts, so a child can never change its parent's variables. In bash, `export NAME=value` puts a variable into the environment of every later child; `NAME=value cmd` sets it for one command only. `${NAME:-default}` reads a variable and substitutes `default` when it is unset or empty. Configuration by environment variable is how the system's services are configured in Kubernetes (`TL_*` variables).

### 2.4 Files, pipes, redirection

Every process starts with **file descriptors** 0 (stdin), 1 (stdout), and 2 (stderr). Programs print results to stdout and diagnostics to stderr, so the two can be separated: `cmd > out.txt 2> err.txt`; `2>&1` sends stderr wherever stdout goes. A **pipe** `a | b` connects a's stdout to b's stdin; both run at once. The status of a pipeline is the status of its **last** command: `grep x missing-file | wc -l` exits 0 even though grep failed, unless `set -o pipefail`, which makes the pipeline fail when any part fails.

### 2.5 The C build model and make

C is built in two stages. **Compiling** turns one `.c` file into one object file: `cc -c greet.c -o greet.o`. A **header** (`greet.h`) declares what another file defines, and `#include "greet.h"` pastes it into every `.c` that uses it. **Linking** combines object files into a program: `cc main.o greet.o -o greet`. Compiling separately is what makes rebuilds cheap: change `greet.c` and only `greet.o` needs compiling again.

**make** automates exactly that decision. A Makefile is a list of **rules**:

```make
target: prerequisite1 prerequisite2
	recipe line run by the shell (it must start with a TAB character)
```

To build a target, make first brings each prerequisite up to date, then runs the recipe **if the target file does not exist or is older than any prerequisite** (by modification time). Plain `make` builds the first target in the file. Pieces the solution uses:

| Piece | Meaning |
|---|---|
| `CC`, `CFLAGS` | variables for the compiler and its flags; `make CC=clang` overrides them from the command line |
| `$@`, `$<`, `$^` | in a recipe: the target, the first prerequisite, all prerequisites |
| `%.o: %.c` | a **pattern rule**: how to make any `.o` from the `.c` of the same name |
| `main.o: greet.h` | a rule with no recipe: adds a prerequisite to `main.o`, because make cannot see `#include` |
| `.PHONY: clean` | `clean` is a name, not a file; run its recipe every time it is asked for |

### 2.6 git

git records **snapshots**. A **commit** is a full snapshot of the tracked files, plus a message, an author, a time, and a pointer to its parent commit, all named by a hash of that content. You choose what goes into the next snapshot by **staging**: `git add file` copies the file's current content into the staging area, and `git commit -m "..."` turns the staging area into a commit. `git status` shows what changed and what is staged; `git log` walks the parents from the newest commit (**HEAD**); `git diff` shows unstaged changes. `.gitignore` lists files git should never offer to track (build outputs like `*.o`, `.venv/`). A **branch** is a movable name for a commit; a **remote** is another copy of the repository you `git push` to. git runs **hooks**, scripts in a hooks directory, at fixed moments: craft.01 uses the `commit-msg` hook, which gets the message file and can reject the commit with a nonzero exit status.

## 3. Worked example by hand

**Who gets rebuilt.** The program has three source files. Both `.c` files include `greet.h`:

```text
greet  <-  main.o  <-  main.c, greet.h
       <-  greet.o <-  greet.c, greet.h
```

Starting from a complete build, make rebuilds exactly the targets downstream of the changed file:

| Changed | Recompiled | Relinked | Why |
|---|---|---|---|
| nothing | nothing | no | every target is newer than its prerequisites |
| `greet.c` | `greet.o` | yes | `greet.o` is now older than `greet.c`; `greet` is then older than `greet.o` |
| `main.c` | `main.o` | yes | same, on the other side |
| `greet.h` | `main.o` and `greet.o` | yes | both list `greet.h` as a prerequisite |
| `greet.h`, header rules forgotten | nothing | no | make never learns that the header matters: stale objects |

**A worker's life.** A session with the reference `worker.sh` (state in `/tmp/w`, a tick every 30 s):

```text
$ WORKER_STATE_DIR=/tmp/w ./worker.sh &        # PID 4242
ready
tick
$ cat /tmp/w/worker.lock
4242
$ kill -TERM 4242                               # trap runs at once: wait returns early
$ wait 4242; echo $?
143                                             # 128 + 15
$ ls /tmp/w/worker.lock
ls: /tmp/w/worker.lock: No such file or directory
```

Exit statuses to recognize: 128 + 2 = **130** (SIGINT), 128 + 9 = **137** (SIGKILL; no trap ran, so a lock file stays behind), 128 + 15 = **143** (SIGTERM).

**The program.** `./greet Ada` prints `hello, Ada` and exits 0; `./greet` with no argument prints `usage: ./greet NAME` on stderr and exits 2.

## 4. The interface

The C program is given and fixed; the exercise is its build. These three files sit in `primers/lang.02/`:

```c
/* primers/lang.02/greet.h */
#ifndef GREET_H
#define GREET_H

#include <stddef.h>

/* Write "hello, <name>" into buf (at most cap bytes, NUL-terminated).
   Returns the length it wanted to write, as snprintf does. */
int greet(char *buf, size_t cap, const char *name);

#endif
```

```c
/* primers/lang.02/greet.c */
#include <stdio.h>

#include "greet.h"

int greet(char *buf, size_t cap, const char *name) {
    return snprintf(buf, cap, "hello, %s", name);
}
```

```c
/* primers/lang.02/main.c: exit codes 0 ok, 1 runtime error, 2 usage error */
#include <stdio.h>

#include "greet.h"

int main(int argc, char **argv) {
    char buf[64];
    if (argc != 2) {
        fprintf(stderr, "usage: %s NAME\n", argv[0]);
        return 2;
    }
    int n = greet(buf, sizeof buf, argv[1]);
    if (n < 0 || (size_t)n >= sizeof buf) {
        fprintf(stderr, "%s: name too long\n", argv[0]);
        return 1;
    }
    puts(buf);
    return 0;
}
```

**The Makefile** (`primers/lang.02/Makefile`) must:

1. build `greet` from `main.o` and `greet.o` on plain `make`, compiling each `.c` to its own `.o` and linking with `$(CC)`;
2. rebuild only what is downstream of a changed file (the table in section 3), headers included;
3. compile and link through `$(CC)`, so `make CC=clang` works;
4. have a `clean` target that removes `greet` and the objects, declared `.PHONY`.

**The worker** (`primers/lang.02/worker.sh`, executable, starting with `#!/usr/bin/env bash`) must:

1. keep its state in `${WORKER_STATE_DIR:-${TMPDIR:-/tmp}}` and tick every `${WORKER_INTERVAL:-30}` seconds;
2. on start, if `worker.lock` exists in the state directory and the PID in it is alive (`kill -0`), print why on stderr and exit 1 without touching the lock; if that PID is gone, remove the stale lock and continue;
3. write its own PID (`$$`) to `worker.lock`, set its traps, then print `ready` as its first line on stdout;
4. on SIGTERM exit 143, on SIGINT exit 130, in both cases after removing the lock and killing any background child, promptly even while it waits out a 30 s interval.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| make builds a program that runs | unit | `make` builds `greet`; `./greet Ada` prints `hello, Ada` with status 0; no argument gives status 2 | the system's C library is built the same way in Pass 1 |
| a second make rebuilds nothing | unit | an up-to-date build does no work | fast edit-build loops |
| editing one source rebuilds only its object | unit | `greet.c` changed: only `greet.o` and the link | incremental builds |
| editing the header rebuilds both objects | boundary | `greet.h` changed: both objects | no stale struct layouts against `tinyllm.h` |
| make uses the `CC` variable | unit | compile and link go through `$(CC)` | sanitizer builds and CI choose the compiler |
| `clean` works even when a file named `clean` exists | boundary | `clean` is `.PHONY` | phony targets always run |
| the worker is an executable script | unit | execute bit and a `#!` line | entry points are exec'd, by CI and by Kubernetes |
| the worker writes its PID to the lock | unit | `worker.lock` holds the worker's PID after `ready` | the section 3 session |
| SIGTERM exits 143 and removes the lock | unit | SIGTERM: status 143, lock gone | graceful shutdown in `L10.0` and `dep.00` |
| SIGINT exits 130 and removes the lock | unit | SIGINT: status 130, lock gone | Ctrl-C behaves too |
| the trap runs promptly during a long sleep | boundary | with a 30 s interval, SIGTERM ends it within 3 s | shutdown inside the grace period |
| no process is left behind | boundary | no process of the worker's group outlives it | no orphans per restart |
| a second start refuses while the first runs | unit | a second start exits 1 and leaves the lock alone | one owner of the lock |
| a stale lock after SIGKILL is recovered | fault | after SIGKILL the lock remains; the next start takes it over | crash recovery without a human |

## 5. The solution, step by step

### 5.1 From first principles

**The Makefile.**

1. make decides "rebuild or not" by comparing file times (2.5). So every rule's target must be a real file the recipe creates; otherwise the target never exists and its recipe always runs.
2. Recompiling only what changed requires one object per source and a separate link step. A pattern rule says this once for every `.c`.
3. make cannot read `#include`. Every object that includes `greet.h` must list it, or a header edit changes nothing (section 3's last row).
4. The compiler is a choice of whoever runs the build (CI, a sanitizer build). So recipes name `$(CC)` and flags are a variable with a default that the caller can override.
5. `clean` is an action, not a file; declaring it `.PHONY` keeps a stray file of that name from making it "up to date".

**The worker.**

6. A lock file proves ownership only if a dead owner can be told apart from a live one. The lock holds a PID, and `kill -0` asks the kernel whether that PID exists, so a lock left by a SIGKILL (which no trap survives) is detected and taken over.
7. A signal must turn into cleanup plus the conventional status 128 + n, so supervisors still see "ended by SIGTERM". That is a trap per signal calling one cleanup function with the right code.
8. A trap runs only between commands (2.2). A 30 s foreground `sleep` would postpone it for up to 30 s, past a container's grace period. Running `sleep` in the background and blocking in `wait` lets the trap run immediately.
9. The backgrounded `sleep` is a separate process. If the trap exits without killing it, it outlives the script as an orphan, one per restart.
10. The lock must be in place before anything announces readiness, and the traps must be set before the lock could be left behind by a signal; `ready` comes last.

### 5.2 The code, annotated

**The Makefile.**

```make
CFLAGS ?= -std=c11 -Wall -Wextra -O2   # ?= : a default the caller can override (step 4)
OBJS := main.o greet.o

.PHONY: all clean                      # names, not files (step 5)
all: greet                             # the first target: plain `make` builds greet

# Link: the program depends on every object.
greet: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

# Compile: one object per source (step 2). Each object also depends on the
# header it includes, so editing greet.h rebuilds both (step 3).
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
main.o: greet.h
greet.o: greet.h

clean:
	rm -f greet $(OBJS)
```

Walk through section 3's table with it. After a full build, `greet` is newer than both objects and each object is newer than its `.c` and `greet.h`: nothing runs. Touch `greet.c`: `greet.o` is older than a prerequisite, so the pattern rule recompiles it (`$<` is `greet.c`, `$@` is `greet.o`); now `greet` is older than `greet.o`, so it relinks. Touch `greet.h`: both objects list it, so both recompile, then the link. The recipe-less rules `main.o: greet.h` only add a prerequisite; the recipe still comes from `%.o: %.c`. `make CC=clang` replaces `CC` everywhere, because no recipe names a compiler. `rm -f` does not fail when the files are already gone, so `make clean` twice in a row succeeds.

**The worker: configuration and the stale-lock check.**

```bash
#!/usr/bin/env bash
set -euo pipefail                      # fail fast: a failing command, unset variable, or pipe stage

state="${WORKER_STATE_DIR:-${TMPDIR:-/tmp}}"   # environment with nested defaults (2.3)
interval="${WORKER_INTERVAL:-30}"
lock="$state/worker.lock"

# A lock whose PID is gone was left by a process that could not clean up
# (SIGKILL, power loss). `kill -0` sends no signal; it only asks "does it exist?".
if [ -e "$lock" ]; then
  old="$(cat "$lock")"
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
    echo "worker: already running (pid $old)" >&2   # diagnostics to stderr
    exit 1                             # and the live owner's lock is left alone
  fi
  echo "worker: removing stale lock (pid ${old:-?})" >&2
  rm -f "$lock"
fi
echo "$$" > "$lock"                    # $$: this script's own PID
```

`kill -0` inside an `if` condition is exempt from `set -e`, so "no such process" is a branch, not a crash. `-n "$old"` guards against an empty lock file, for which `kill -0 ""` would be an error rather than a no.

**The worker: the trap.**

```bash
stop() {
  # Kill the background sleep too, or it outlives us as an orphan. `jobs -p`
  # lists every background child, including one started an instant before
  # the signal arrived.
  local kids
  kids="$(jobs -p)"
  # shellcheck disable=SC2086  # one word per PID: the splitting is the point
  if [ -n "$kids" ]; then kill $kids 2>/dev/null || true; fi   # the child may already be gone
  rm -f "$lock"
  exit "$1"                            # 128 + n, chosen by the trap (step 7)
}
trap 'stop 143' TERM   # 128 + 15
trap 'stop 130' INT    # 128 + 2
```

One function, two traps: the cleanup is identical and only the status differs. `|| true` keeps `set -e` from aborting the cleanup halfway if the `sleep` exited on its own an instant earlier. The lock is removed before `exit`, so a supervisor that restarts the worker on exit finds no lock.

**The worker: the loop.**

```bash
echo ready                             # only now: lock written, traps set (step 10)
while true; do
  echo tick
  # Sleep in the background and `wait` for it: bash runs a trap only between
  # commands, and `wait` returns as soon as a trapped signal arrives, while a
  # foreground `sleep 30` would delay the trap for up to 30 seconds.
  sleep "$interval" &
  wait $! || true
done
```

Section 3's session follows line by line: `ready` and `tick` are printed; `kill -TERM 4242` interrupts `wait` (step 8), bash runs `stop 143`, which kills the `sleep`, removes `/tmp/w/worker.lock`, and exits 143. When a signal interrupts `wait`, bash runs the trap, which exits, so the loop never continues. `|| true` covers the other case: a `sleep` that fails for some other reason (a bad `WORKER_INTERVAL`), which under `set -e` would otherwise end the worker with its lock still in place, since there is no trap on plain exit.

Full solution: [`Makefile`](../../../../course/solutions/primers/lang.02/Makefile), [`worker.sh`](../../../../course/solutions/primers/lang.02/worker.sh).

### 5.3 Why this solution

- **Cost.** An up-to-date `make` does one `stat` per file and runs nothing; a one-file change runs one compile and one link. The worker is idle in `wait` between ticks and reacts to a signal in milliseconds.
- **Explicit header rules** rather than `-MMD -MP` dependency files: for two sources and one header they are clearer, and they show what the generated files would contain. A project with many headers should let the compiler generate them (see Going further).
- **A PID lock with `kill -0`** rather than `flock`: it is portable shell and makes the stale-lock problem visible. Its weakness is a race: two workers starting at the same instant can both see no lock and both write one. `flock` closes that race because the kernel owns the lock and drops it when the process dies.
- **Tradeoff accepted:** PID reuse. If the dead owner's PID has been reused by an unrelated process, `kill -0` says "alive" and the worker refuses to start. That fails safe (no two owners) but needs a human; a kernel lock avoids it.

### 5.4 Explain it back

**Why is there an `all` target instead of making `greet` the first rule?**

<details><summary>Answer</summary>

Either works for plain `make`, which builds the first target. `all` gives a stable name for "build everything" that stays correct if more programs are added, and declaring it `.PHONY` means its own up-to-date state never matters: only `greet`'s does.

</details>

**What does `make` do after `touch greet.h`, and which lines of the Makefile cause it?**

<details><summary>Answer</summary>

It recompiles `main.o` and `greet.o` and relinks `greet`. `main.o: greet.h` and `greet.o: greet.h` make both objects older than a prerequisite; the pattern rule supplies their recipe; then `greet: $(OBJS)` is older than its objects.

</details>

**The worker is killed with SIGKILL. What is left behind, and how does the next start cope?**

<details><summary>Answer</summary>

No trap runs, so `worker.lock` stays with the dead PID, and the backgrounded `sleep` may live on until it finishes its interval. The next start reads the PID, `kill -0` fails, so it removes the stale lock and takes over.

</details>

**Why is `sleep` run in the background and then waited for, rather than in the foreground?**

<details><summary>Answer</summary>

bash runs a trap only after the current foreground command finishes, so a SIGTERM during `sleep 30` would be handled up to 30 s later. The builtin `wait` returns as soon as a trapped signal arrives, so the trap runs at once.

</details>

**Why exit 143 instead of 0 after a clean SIGTERM shutdown?**

<details><summary>Answer</summary>

128 + 15 tells whoever waits on the process that it ended because of SIGTERM, which is the convention shells and supervisors use. Exiting 0 would claim the work finished on its own, hiding that it was stopped.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. A rule whose target never exists as a file (for example `build:` producing `greet`) | every `make` relinks or recompiles everything | the link rule's target is the file `greet` itself; `all` is `.PHONY` and only depends on it |
| 2. One rule `cc main.c greet.c -o greet` | every change recompiles every file | the pattern rule `%.o: %.c` compiles each source to its own object |
| 3. No `main.o: greet.h` and `greet.o: greet.h` | after a header change the objects are stale and disagree with it | `main.o: greet.h` and `greet.o: greet.h` |
| 4. A recipe that hardcodes `gcc` | `make CC=clang` and sanitizer builds are ignored | every recipe runs `$(CC)` |
| 5. `clean` not declared `.PHONY` | a stray file called `clean` makes `make clean` do nothing | `.PHONY: all clean` |
| 6. No trap | the shell dies with the signal (status shown as -15 by Python), the lock stays | `trap 'stop 143' TERM`; `stop` removes the lock |
| 7. A foreground `sleep "$interval"` | SIGTERM waits up to 30 s for the sleep to finish | `sleep "$interval" &` then `wait $!` |
| 8. The trap forgets the background `sleep` | one orphaned process per restart | `stop` kills every PID from `jobs -p` |
| 9. Trusting any existing lock | after one SIGKILL the worker never starts again | the start checks the lock's PID with `kill -0` and removes a stale lock |
| 10. Spaces instead of a TAB before a recipe line | `Makefile:9: *** missing separator.  Stop.` | recipe lines start with a TAB |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Forward | `craft.01` | CI jobs are processes whose exit status is the gate; the `commit-msg` hook is a script that rejects with a nonzero status |
| Forward | `M03.1` | the C exercise's Makefile builds standalone test binaries from source files and headers |
| Forward | `rt.01` | the same Makefile gains `SANITIZE=1` (a variable, 2.5) to add AddressSanitizer |
| Forward | `L10.0` | the Rust engine handles SIGTERM: stop accepting, finish in-flight streams, exit |
| Forward | `dep.00` | Kubernetes sends SIGTERM, waits the grace period (30 s by default), then SIGKILL; status 137 in `kubectl describe` means SIGKILL |
| Forward | `ops.00` | the first drill reads a crashlooping pod's exit status and restart count |

A primer has no code call site. The Pass 0 local CI script builds and cleans this primer (`make -C primers/lang.02`) as a toolchain check, so a broken compiler or make fails at the start.

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| header rules in the `Makefile` | `cc -MMD -MP` dependency files, Ninja | the compiler writes the header list itself; a faster scheduler | GNU Make manual, "Generating Prerequisites Automatically" |
| the worker's lock file | `flock(1)`, `fcntl` locks | the kernel releases the lock when the process dies, so there is no stale lock and no check-then-write race | `man 1 flock` (util-linux) |
| the TERM trap | `tini`, `docker run --init` | a minimal PID 1 that forwards signals and reaps orphans inside a container | krallin/tini `README.md` |
| exit 143 on SIGTERM | Kubernetes pod termination | `preStop` hooks, `terminationGracePeriodSeconds`, readiness removal before SIGTERM | kubernetes.io, "Pod Lifecycle", termination of Pods |
