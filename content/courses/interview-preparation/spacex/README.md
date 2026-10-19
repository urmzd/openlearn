# Interviewing at SpaceX

SpaceX is one of the most selective software employers anywhere. Its process runs seven to nine rounds, the reported pass rate leaves the "vast majority" of candidates without an offer, and the software you would be interviewing to write is mission-critical in the literal sense: it flies rockets and operates the Starlink constellation. That last fact shapes everything about the interviews. A bug in a web app costs revenue; a bug in flight software can destroy a vehicle or hurt people, so the interviewers look for correctness, fault tolerance, and a habit of reasoning about the physical world, not just algorithmic speed.

This chapter walks through the process in the order you will meet it. It starts with the shape of the pipeline and the compensation tradeoff, then the themes that run through every round. The middle of the chapter is technical: the four-hour take-home, the onsite algorithms and systems-programming rounds (with three worked listings you should be able to explain line by line), and the system design topics that are specific to SpaceX. It closes with the behavioral and hiring-manager round, where AI and ML show up in the company, and a preparation strategy.

> **Learning objectives**
> - Describe the SpaceX interview pipeline, its length, and what each round tests.
> - Explain and extend the three reference solutions: satellite visibility by elevation angle, launch-window intersection with a sweep line, and a binary telemetry parser with sync-word search and CRC-16 validation.
> - Sketch a fault-tolerant design for a Starlink ground network, a flight software stack, or constellation management, and name the mechanisms (TMR, watchdogs, graceful degradation) that make each safe.
> - Answer first-principles and mission questions honestly and specifically.

## Before you begin

The coding rounds assume fluent data structures and algorithms; the [Algorithms](../../algorithms/README.md) book covers the sweep-line and graph-search patterns used here. The system design round draws on the [Distributed Systems](../shared-concepts/distributed-systems.md) chapter of this book, and the systems-programming round rewards comfort with byte layouts and bit manipulation. Basic trigonometry is enough for the geometry; [Mathematics for Computing](../../mathematics-for-computing/README.md) has a refresher if you need one.

## In this chapter

- [The interview process](#the-interview-process): the nine-round pipeline and how long it takes.
- [Compensation](#compensation): why the cash is lower than Big Tech and what the equity offers in return.
- [What SpaceX is looking for](#what-spacex-is-looking-for): the six themes that every round probes.
- [The take-home assessment](#the-take-home-assessment): format, reported problem types, and a worked satellite-coverage solution.
- [The onsite coding rounds](#the-onsite-coding-rounds): a launch-window sweep line and a binary telemetry parser.
- [The system design round](#the-system-design-round): Starlink ground stations, flight software, and constellation management.
- [The behavioral and hiring manager round](#the-behavioral-and-hiring-manager-round): culture, common questions, and first-principles reasoning.
- [AI and ML at SpaceX](#ai-and-ml-at-spacex) and [Preparing well](#preparing-well): where ML is used and how to spend your preparation time.

## The interview process

Expect the whole process to take four to eight weeks, which makes it one of the longest in the industry. The rounds, in order, look like this:

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, motivation, clearance |
| Take-Home Assessment | Coding | 4 hours | Hard algorithms problem |
| Phone Screen 1 | Technical | 60 min | DS&A, systems |
| Phone Screen 2 | Technical | 60 min | Domain-specific (embedded, web, infra) |
| Onsite 1 | Coding | 60 min | Algorithms |
| Onsite 2 | Coding | 60 min | Systems programming |
| Onsite 3 | System Design | 60 min | Mission-critical architecture |
| Onsite 4 | Domain Deep Dive | 60 min | Team-specific technical |
| Onsite 5 | Hiring Manager | 45 min | Culture, mission, leadership |

Candidates rate the difficulty 3.4 out of 5 on Glassdoor, which is high for any company. Note that the recruiter screen already asks about clearance: much of SpaceX's work falls under export-control rules, so eligibility comes up early.

## Compensation

The offer is structured differently from Big Tech. For a senior engineer in the US, reported base salary runs \$160-200K, the annual bonus is minimal, and the rest is SpaceX stock options in a pre-IPO company. Total compensation lands around \$300-500K. In cash terms that is below the large public tech companies; the equity is what makes up the difference, and historically it has been extremely valuable. Before you interview, decide how much of your pay you are comfortable holding as illiquid upside, because the hiring manager may ask.

## What SpaceX is looking for

Six themes recur across the rounds, and it helps to know them before you sit down for the first one.

The first is the mission. SpaceX engineers are motivated by the goal of making life multi-planetary, and genuine enthusiasm for space is expected rather than optional. The second follows from the domain: the software is mission-critical, so correctness, reliability, and fault tolerance outrank cleverness. Third, the work is full-stack in the physical world. Code controls rockets, satellites, ground stations, and manufacturing robots, and an answer that ignores sensors, actuators, or radio links will read as a web-app answer.

Fourth, the work is intense. SpaceX is known for 50-60+ hour weeks, and you should be ready to talk honestly about how you relate to that pace. Fifth, the languages are C++ and Python: flight software is C++, while ground systems, tooling, and ML are Python, and both matter. Finally, many roles are embedded and distributed at once. Starlink puts embedded systems on every satellite and also runs a massive distributed ground infrastructure, so you may be asked about both ends.

## The take-home assessment

The take-home is a single hard problem with a four-hour clock that starts when you open it. It is usually done in C++ or Python and submitted by email or through an online platform. Because you have time, the bar is higher than in a live round: graders expect clean, documented code, and they look closely at your test cases and edge-case handling, not just at whether the main path works.

Reported problems tend to model something real: path planning (A* or Dijkstra with constraints), simulation of a physical system, resource allocation and scheduling optimization, or signal processing and data parsing. The listing below is representative. Given satellites at known positions and altitudes and a set of ground stations, it reports which stations each satellite can talk to.

The key idea is the elevation angle: how far above the horizon the satellite appears from the ground station. Below a minimum elevation (25° here, a typical value for Starlink terminals) the signal passes through too much atmosphere and terrain clutter to be useful. To compute it, first find the central angle $\gamma$ between the station and the sub-satellite point with the haversine formula, which stays numerically stable for small angles:

$$a = \sin^2\!\left(\tfrac{\Delta\phi}{2}\right) + \cos\phi_1 \cos\phi_2 \sin^2\!\left(\tfrac{\Delta\lambda}{2}\right), \qquad \gamma = 2\arcsin\sqrt{a}.$$

With Earth radius $R$ and satellite orbital radius $r = R + h$, the elevation angle $\varepsilon$ satisfies

$$\tan\varepsilon = \frac{\cos\gamma - R/r}{\sin\gamma},$$

which the code evaluates with `atan2` so that the quadrant is handled correctly.

```python
# Example: Satellite coverage optimization
from typing import List, Tuple
import math

def max_ground_coverage(
    satellites: List[Tuple[float, float, float]],  # (lat, lon, altitude_km)
    ground_stations: List[Tuple[float, float]],     # (lat, lon)
    min_elevation_deg: float = 25.0
) -> List[List[int]]:
    """For each satellite, find which ground stations it can communicate with.

    A satellite can see a ground station if the elevation angle from the
    ground station to the satellite exceeds min_elevation_deg.
    """
    EARTH_RADIUS_KM = 6371.0

    def elevation_angle(sat_lat, sat_lon, sat_alt, gs_lat, gs_lon) -> float:
        """Compute elevation angle from ground station to satellite."""
        # Convert to radians
        lat1 = math.radians(gs_lat)
        lon1 = math.radians(gs_lon)
        lat2 = math.radians(sat_lat)
        lon2 = math.radians(sat_lon)

        # Central angle using Haversine
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        central_angle = 2 * math.asin(math.sqrt(a))

        # Elevation angle
        sat_distance = EARTH_RADIUS_KM + sat_alt
        elevation = math.atan2(
            math.cos(central_angle) - EARTH_RADIUS_KM / sat_distance,
            math.sin(central_angle)
        )
        return math.degrees(elevation)

    coverage = []
    for sat_lat, sat_lon, sat_alt in satellites:
        visible = []
        for gs_idx, (gs_lat, gs_lon) in enumerate(ground_stations):
            elev = elevation_angle(sat_lat, sat_lon, sat_alt, gs_lat, gs_lon)
            if elev >= min_elevation_deg:
                visible.append(gs_idx)
        coverage.append(visible)

    return coverage
```

The solution is a direct $O(S \cdot G)$ double loop over satellites and stations, which is the right first answer. In a take-home you would then add what the graders look for: tests for a satellite directly overhead (elevation 90°), one exactly on the threshold, and stations near the antimeridian where longitudes wrap. If the follow-up asks for thousands of satellites and stations, the natural improvement is to bucket stations by a coarse latitude/longitude grid and only test stations within the maximum ground radius a satellite at that altitude can cover.

## The onsite coding rounds

The onsite has two coding rounds. One is standard data structures and algorithms with an engineering flavor; the other is systems programming.

### Algorithms: finding launch windows

A typical algorithms problem wraps a classic pattern in a launch scenario. Each constraint (weather, range availability, orbital alignment) is open during a window `(start, end)`, and you need every interval in which all constraints hold at once for at least `duration` time units.

The sweep-line technique solves it. Turn each window into two events, $+1$ at its start and $-1$ at its end, and sort them by time. Walking the events in order while keeping a running count of active constraints tells you, at every moment, how many are satisfied. When the count reaches $n$ you have entered an all-satisfied interval; when it drops below $n$ you have left it, and you keep the interval if it was long enough.

```python
# Scheduling with constraints (rocket launch windows)
def find_launch_windows(
    constraints: list[tuple[int, int]],  # (start, end) windows for each constraint
    duration: int
) -> list[tuple[int, int]]:
    """Find time windows where ALL constraints are satisfied for at least `duration`."""
    events = []
    for start, end in constraints:
        events.append((start, 1))
        events.append((end, -1))
    events.sort()

    n = len(constraints)
    active = 0
    all_satisfied_start = None
    windows = []

    for time, delta in events:
        active += delta
        if active == n and all_satisfied_start is None:
            all_satisfied_start = time
        elif active < n and all_satisfied_start is not None:
            if time - all_satisfied_start >= duration:
                windows.append((all_satisfied_start, time))
            all_satisfied_start = None

    return windows
```

Sorting dominates, so the run time is $O(n \log n)$. Two details are worth saying out loud in the interview. First, because tuples sort by time and then by delta, an end event $(t, -1)$ comes before a start event $(t, +1)$ at the same time, which treats windows as half-open: two constraints that merely touch at $t$ do not count as overlapping. Second, the code assumes each constraint contributes exactly one window, since it compares against `n = len(constraints)`. If a constraint can have several windows, you would need to track per-constraint activity (or merge each constraint's windows first) rather than a single count.

### Systems programming: parsing binary telemetry

The systems round often asks you to parse a binary protocol, and satellite telemetry is a natural framing. Packets arrive in a byte stream that may contain noise or partial packets, so the parser must find the start of each packet, decode fixed-width fields in the right byte order, and reject anything corrupted.

The packet layout is big-endian and 46 bytes long:

| Bytes | Field | Type |
|------:|-------|------|
| 4 | sync word (`0xDEADBEEF`) | uint32 |
| 8 | timestamp (microseconds) | uint64 |
| 2 | satellite ID | uint16 |
| 4 | battery voltage | float32 |
| 4 | temperature | float32 |
| 8 | GPS latitude | float64 |
| 8 | GPS longitude | float64 |
| 4 | GPS altitude | float32 |
| 2 | status flags | uint16 |
| 2 | CRC-16 | uint16 |

The parser scans byte by byte for the sync word. When it finds one, it decodes the payload with `struct` and checks the CRC over the first 44 bytes (everything but the CRC itself). A valid packet is emitted and the cursor jumps a full packet ahead; a sync word that fails the CRC is treated as a false match (four payload bytes can happen to equal `0xDEADBEEF`), and the scan resumes one byte later. The `struct` format string must mirror the table exactly: `>` for big-endian, then `Q H f f d d f H H` for timestamp, ID, voltage, temperature, latitude, longitude, altitude, flags, and CRC.

```python
# Binary protocol parser (satellite telemetry)
import struct
from dataclasses import dataclass

@dataclass
class TelemetryPacket:
    timestamp_us: int
    satellite_id: int
    battery_voltage: float
    temperature_c: float
    gps_lat: float
    gps_lon: float
    gps_alt: float
    status_flags: int

class TelemetryParser:
    """Parse binary telemetry packets from satellites.

    Packet format (big-endian):
    - 4 bytes: sync word (0xDEADBEEF)
    - 8 bytes: timestamp (uint64, microseconds)
    - 2 bytes: satellite ID (uint16)
    - 4 bytes: battery voltage (float32)
    - 4 bytes: temperature (float32)
    - 8 bytes: GPS latitude (float64)
    - 8 bytes: GPS longitude (float64)
    - 4 bytes: GPS altitude (float32)
    - 2 bytes: status flags (uint16)
    - 2 bytes: CRC-16
    Total: 46 bytes
    """

    SYNC_WORD = 0xDEADBEEF
    PACKET_SIZE = 46
    HEADER_FMT = ">I"           # sync word
    PAYLOAD_FMT = ">QHffddfHH"  # payload + CRC (42 bytes)

    def parse_stream(self, data: bytes) -> list[TelemetryPacket]:
        """Parse a byte stream, finding and extracting valid packets."""
        packets = []
        i = 0
        while i <= len(data) - self.PACKET_SIZE:
            # Find sync word
            sync = struct.unpack_from(self.HEADER_FMT, data, i)[0]
            if sync != self.SYNC_WORD:
                i += 1
                continue

            # Parse payload
            payload = struct.unpack_from(self.PAYLOAD_FMT, data, i + 4)
            ts, sat_id, voltage, temp, lat, lon, alt, flags, crc = payload

            # Verify CRC
            if self._crc16(data[i:i + self.PACKET_SIZE - 2]) == crc:
                packets.append(TelemetryPacket(
                    timestamp_us=ts, satellite_id=sat_id,
                    battery_voltage=voltage, temperature_c=temp,
                    gps_lat=lat, gps_lon=lon, gps_alt=alt,
                    status_flags=flags
                ))
                i += self.PACKET_SIZE
            else:
                i += 1  # CRC failed, try next byte

        return packets

    def _crc16(self, data: bytes) -> int:
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc
```

The CRC is the common CRC-16/MODBUS variant: start from `0xFFFF`, XOR each byte into the low bits, and for each of the eight bits shift right, XORing in the reflected polynomial `0xA001` whenever the bit shifted out was 1.

> **Warning:** The format string is the easiest place to lose the round. An earlier version of this listing used `">QHffddHH"`, which drops the float32 altitude. That string decodes only 38 bytes into eight values, so unpacking it into nine names raises `ValueError` on the first real packet. Always add up your format against the layout: $8+2+4+4+8+8+4+2+2 = 42$ payload bytes plus the 4-byte sync word gives the 46-byte packet. `struct.calcsize` checks this for you in one line.

Good follow-ups to anticipate: a stream that ends mid-packet (the loop bound leaves the tail unparsed, so a streaming version must buffer it for the next read), a CRC table lookup to replace the bit-by-bit loop when throughput matters, and how you would write the same parser in C++ without dynamic allocation.

## The system design round

The design round is about mission-critical architecture. Interviewers want to see you plan for failure first and features second. Three SpaceX-specific prompts come up repeatedly.

### Designing the Starlink ground station network

Traffic flows from satellites in low Earth orbit down to ground station antennas, through gateway routers and a traffic engineering layer, and out to the internet, peering partners, points of presence, and enterprise customers:

```
[Satellites (LEO)] --> [Ground Station Antennas]
                              |
                        [Gateway Routers]
                              |
                        [Traffic Engineering]
                              |
                  +--------+--------+--------+
                  |        |        |        |
            [Internet]  [Peering]  [PoP]  [Enterprise]
```

The interesting problems all come from motion and failure. Satellites move overhead in minutes, so connections must be handed off seamlessly from one satellite to the next. Laser inter-satellite links let traffic travel long distances in space without bouncing through ground hops. Low orbit is what makes the latency competitive: roughly 20-40 ms round trip, against about 600 ms for a geostationary link. Capacity planning follows geography, since each satellite serves an area and user density there drives the capacity you need. And everything fails: satellites, ground stations, and links. The design must degrade gracefully when any of them do.

### Designing a flight software architecture

Flight software is a control loop with safety systems wrapped around every stage. Sensors feed a state estimator, which feeds guidance, then control, then the actuators; redundancy, mission logic, and safety systems sit alongside:

```
[Sensors] --> [State Estimator] --> [Guidance] --> [Control] --> [Actuators]
                   |                    |               |
              [Redundancy]        [Mission Logic]  [Safety System]
                   |                    |               |
              [Voting (TMR)]     [Autonomous Abort]  [Hardware Limits]
```

The mechanisms to name are these. Triple modular redundancy (TMR) runs three computers that vote on every decision, and disagreement detection triggers a fallback when one of them diverges. Execution must be deterministic: no dynamic allocation, no garbage collection, and no unbounded loops, so that worst-case timing is known. Watchdog timers detect a hung process and trigger a reset. And because cosmic rays flip bits in memory, the software must be designed to tolerate radiation-induced faults rather than assume the hardware is perfect.

### Designing satellite constellation management

The third prompt is operating thousands of satellites as a fleet. You need to track the orbital positions of 5,000+ satellites, predict conjunctions and execute collision-avoidance maneuvers, schedule end-of-life deorbits, push software updates safely to thousands of spacecraft already in orbit, and detect satellites whose performance is degrading before they fail. A strong answer treats the software rollout like a staged deployment with health checks and rollback, because a bad update to a constellation cannot be fixed by walking over to the machine.

## The behavioral and hiring manager round

The final round tests fit with a demanding culture. SpaceX values mission obsession ("Are you excited to make life multi-planetary?"), first-principles thinking (a favorite framework of Elon Musk's), extreme ownership of your system end to end, and a bias for action: move fast, iterate, and test in hardware. It also values work ethic, and SpaceX works very hard. Be honest about your comfort with that; pretending will not survive the follow-up questions.

The common questions are predictable, which means you can prepare real stories for them:

- "Why SpaceX?" (This must be genuine and specific.)
- "Tell me about the hardest technical problem you've solved."
- "Describe a time you had to make a critical decision under time pressure."
- "How do you handle failure? Give a specific example."
- "What would you do if you found a software bug minutes before a launch?"

SpaceX may also ask you to reason from first principles about an unfamiliar question. The point is to decompose the problem into physical facts rather than recall an answer. Why is a rocket cylindrical? Because a cylinder is a structurally efficient pressure vessel. How would you estimate the bandwidth Starlink needs? Multiply users by per-user throughput by utilization. Why is testing harder for spacecraft? Because you cannot reproduce the environment, radiation introduces faults, and once launched you have no physical access.

## AI and ML at SpaceX

ML roles and ML-flavored questions are growing. Computer vision supports autonomous landing by detecting the landing pad and feeding precision guidance. Starlink uses ML for traffic routing and capacity management. Manufacturing uses vision for quality inspection, detecting anomalies in welds and heat shields. Telemetry feeds anomaly-detection models that predict component failures on satellites and rockets, and reinforcement learning is used for fuel-optimal trajectory optimization. If you are interviewing for one of these teams, expect the design round to ask how you would make such a model safe to act on in a system that cannot tolerate a wrong answer.

## Preparing well

The best preparation follows the themes above. Start with honesty about the mission: interviewers detect faked enthusiasm, and if space does not excite you, SpaceX is probably not the right fit. Then build depth where the rounds go deep. Practice systems programming: binary protocols, embedded-systems patterns, and real-time constraints. For flight software roles, know C++ well enough to discuss deterministic memory, code without exceptions, and avoiding dynamic allocation on hot paths. Be fluent in fault-tolerance patterns: TMR and voting, graceful degradation, and watchdog timers. For Starlink roles, study routing protocols, mesh networking, and handoff algorithms.

Learn enough physics to be credible: orbital mechanics, signal propagation, and atmospheric effects. You do not need a PhD, but understanding the domain changes the quality of your answers. Pace yourself across seven to nine rounds, which is exhausting, and go in having already decided how you feel about the compensation tradeoff of lower cash against potentially large equity upside.

## Key terms

- **Elevation angle**: the angle of a satellite above the local horizon as seen from a ground station; links are usable only above a minimum value.
- **Sweep line**: an algorithm that converts intervals into sorted start and end events and processes them in order while maintaining a running state.
- **Sync word**: a fixed bit pattern that marks the start of a packet in a byte stream.
- **CRC-16**: a 16-bit cyclic redundancy check used to detect corrupted packets.
- **Triple modular redundancy (TMR)**: running three independent computers and taking a majority vote on each output.
- **Watchdog timer**: a hardware or software timer that resets a system that stops checking in.
- **LEO**: low Earth orbit, which gives Starlink its 20-40 ms round-trip latency.

## Summary

- SpaceX runs one of the longest processes in tech: seven to nine rounds over four to eight weeks, including a four-hour take-home.
- Compensation trades lower cash (about \$300-500K total for seniors) for pre-IPO equity upside.
- Coding problems wrap classic patterns (geometry, sweep lines, binary parsing) in physical scenarios, and graders care about edge cases and tests.
- System design is judged on fault tolerance: handoff and graceful degradation for Starlink, TMR, determinism, and watchdogs for flight software.
- The behavioral round looks for genuine mission motivation, ownership, honesty about intensity, and first-principles reasoning.

## Review questions

1. Why does the satellite-coverage solution use `atan2` rather than `atan` to compute the elevation angle?
   <details><summary>Answer</summary>`atan2(y, x)` uses the signs of both arguments to return the correct quadrant and stays defined when $\sin\gamma$ is zero (satellite directly overhead). With `atan(y/x)` you would divide by zero overhead and lose the sign information that distinguishes a satellite below the horizon.</details>
2. In `find_launch_windows`, what happens when one constraint ends at exactly the time another begins, and why?
   <details><summary>Answer</summary>They are not treated as overlapping. Events are tuples sorted by `(time, delta)`, so the end event `(t, -1)` sorts before the start event `(t, +1)`. The active count drops before it rises, so the instant $t$ is never counted as all-satisfied: windows behave as half-open intervals.</details>
3. What is wrong with the format string `">QHffddHH"` for the 46-byte telemetry packet, and how would you catch it?
   <details><summary>Answer</summary>It omits the float32 GPS altitude, so it describes 38 bytes and yields eight values, while the code unpacks nine and raises `ValueError`. The correct string is `">QHffddfHH"` (42 bytes). Checking `struct.calcsize(fmt) + 4 == PACKET_SIZE` in a test, or asserting it at class definition, catches the mistake immediately.</details>
4. Why does the parser advance by one byte, not a full packet, when the CRC fails?
   <details><summary>Answer</summary>A CRC failure means the sync word was probably a false match inside other data, or the packet was corrupted. Skipping 46 bytes could jump over the start of a real packet; advancing one byte resumes the search without losing any candidate sync word.</details>
5. Name three properties flight software must have that most web services do not, and explain why each matters.
   <details><summary>Answer</summary>Deterministic execution (no dynamic allocation, no garbage collection, no unbounded loops) so worst-case timing is known; redundancy with voting (TMR) so a single faulty computer cannot command the vehicle; and fault detection with watchdog timers and radiation tolerance so hangs and bit flips are caught and recovered automatically, since nobody can reboot the machine by hand in flight.</details>
6. How would you estimate Starlink's bandwidth needs from first principles?
   <details><summary>Answer</summary>Multiply the number of users by the throughput each needs and by the utilization (the fraction actively using it at peak), then distribute that over geography, since each satellite serves an area and user density determines how much capacity each area requires.</details>

## Further reading

- [SpaceX Software Engineer Interview Questions - InterviewQuery](https://www.interviewquery.com/interview-guides/spacex-software-engineer)
- [SpaceX Interview Process - 4dayweek.io](https://4dayweek.io/interview-process/spacex)
- [Getting a Job Offer at SpaceX - InterviewPal](https://www.interviewpal.com/blog/getting-a-job-offer-at-spacex-interview-process-and-top-questions-to-practice)
- [SpaceX's Interview Process - interviewing.io](https://interviewing.io/spacex-interview-questions)
- [Glassdoor - SpaceX SWE Interview Questions](https://www.glassdoor.com/Interview/SpaceX-Software-Engineer-Interview-Questions-EI_IE40371.0,6_KO7,24.htm)
- [Jointaro - SpaceX Interview Experiences](https://www.jointaro.com/interviews/companies/spacex/)
- Community reports on r/SpaceX, r/cscareerquestions, and Blind
