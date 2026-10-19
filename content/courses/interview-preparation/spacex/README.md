# SpaceX Software Engineer Interview Guide

Comprehensive preparation for SpaceX SWE roles. SpaceX is **extremely selective**, with 7-9 interview rounds and a "vast majority" of candidates failing. The software is mission-critical -- it flies rockets and operates Starlink.

## Interview Process Overview

Timeline: **4-8 weeks**, **7-9 rounds** (one of the longest processes in tech)

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

Interview difficulty rated **3.4/5** on Glassdoor (high for any company).

## Compensation (Senior, US)

- **Base**: $160-200K
- **Equity**: SpaceX stock options (pre-IPO, significant potential upside)
- **Bonus**: Minimal
- **Total Comp**: ~$300-500K (lower cash than Big Tech, compensated by equity upside)
- SpaceX pays below Big Tech in cash but equity has been extremely valuable

## Key Themes

1. **Mission above all** -- SpaceX engineers are deeply motivated by the mission to make life multi-planetary. Genuine passion for space is expected, not optional.
2. **Mission-critical software** -- Software at SpaceX can kill people if it fails. Correctness, reliability, and fault tolerance are paramount.
3. **Full-stack in the physical world** -- Software controls rockets, satellites, ground stations, and manufacturing robots. It's not web apps.
4. **Work intensity** -- SpaceX is known for demanding hours (50-60+ per week). Be prepared to discuss your relationship with intense work.
5. **C++ and Python** -- Flight software is C++. Ground systems, tooling, and ML are Python. Both are critical.
6. **Embedded + distributed** -- Starlink involves embedded systems on satellites AND massive distributed ground infrastructure.

## Take-Home Assessment

### Format

- **Duration**: 4 hours (timed from when you start)
- **Difficulty**: Hard
- **Language**: Usually C++ or Python
- **Submitted**: Via email or online platform

### What to Expect

- A single complex algorithmic problem
- May involve simulation, optimization, or real-world modeling
- Clean, well-documented code expected
- Test cases and edge case handling matter

### Reported Problem Types

- Path planning algorithms (A*, Dijkstra with constraints)
- Simulation of physical systems
- Resource allocation and scheduling optimization
- Signal processing / data parsing

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

## Coding Rounds (Onsite)

### Algorithms

Standard DS&A but with a systems/engineering flavor:

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

### Systems Programming

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
    HEADER_FMT = ">I"         # sync word
    PAYLOAD_FMT = ">QHffddHH"  # payload + CRC

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

## System Design Round

### SpaceX-Specific Topics

#### Design Starlink Ground Station Network

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

- **Satellite handoff**: As satellites move overhead, seamlessly transfer connections
- **Routing**: Laser inter-satellite links for long-distance routing without ground hops
- **Latency**: ~20-40ms LEO round trip vs. ~600ms GEO
- **Capacity planning**: Each satellite serves a geographic area; density drives capacity
- **Fault tolerance**: Satellites fail; ground stations fail; links fail. Everything must degrade gracefully.

#### Design Flight Software Architecture

```
[Sensors] --> [State Estimator] --> [Guidance] --> [Control] --> [Actuators]
                   |                    |               |
              [Redundancy]        [Mission Logic]  [Safety System]
                   |                    |               |
              [Voting (TMR)]     [Autonomous Abort]  [Hardware Limits]
```

- **Triple modular redundancy (TMR)**: Three computers vote on every decision
- **Fault detection**: Disagree detection triggers fallback
- **Deterministic execution**: No dynamic allocation, no garbage collection, no unbounded loops
- **Watchdog timers**: Detect hung processes, trigger reset
- **Radiation hardening**: Software must handle bit flips from cosmic rays

#### Design Satellite Constellation Management

- **Orbital mechanics**: Track positions of 5,000+ satellites
- **Collision avoidance**: Predict and execute avoidance maneuvers
- **Deorbit planning**: Schedule end-of-life deorbits
- **Software updates**: Push updates to thousands of satellites in orbit
- **Anomaly detection**: Identify satellites with degraded performance

## Behavioral / Hiring Manager Round

### SpaceX Culture

- **Mission obsession**: "Are you excited to make life multi-planetary?"
- **First principles thinking**: Elon Musk's favorite question framework
- **Extreme ownership**: You own your system end-to-end
- **Bias for action**: Move fast, iterate, test in hardware
- **Work ethic**: SpaceX works hard. Very hard. Be honest about your comfort with this.

### Common Questions

- "Why SpaceX?" (Must be genuine and specific)
- "Tell me about the hardest technical problem you've solved."
- "Describe a time you had to make a critical decision under time pressure."
- "How do you handle failure? Give a specific example."
- "What would you do if you found a software bug minutes before a launch?"

### The "First Principles" Question

SpaceX may ask you to reason from first principles about a problem:

- "Why does a rocket have a cylindrical shape?" (Structural efficiency, pressure vessel)
- "How would you estimate the bandwidth needed for Starlink?" (Users x throughput x utilization)
- "Why is software testing harder for spacecraft?" (Can't reproduce environment, radiation, no physical access)

## AI/ML at SpaceX

- **Autonomous landing**: Computer vision for landing pad detection and precision guidance
- **Starlink traffic optimization**: ML for routing and capacity management
- **Manufacturing**: Computer vision for quality inspection (detecting anomalies in welds, heat shields)
- **Anomaly detection**: ML on telemetry data to predict satellite/rocket component failures
- **Trajectory optimization**: Reinforcement learning for fuel-optimal trajectories

## Preparation Tips

1. **Genuine space passion** -- They will detect faking. If you're not excited about space, SpaceX isn't the right fit.
2. **Systems programming** -- Binary protocols, embedded systems patterns, real-time constraints.
3. **C++ depth** -- For flight software roles: deterministic memory, no exceptions, no dynamic allocation in hot paths.
4. **Fault tolerance** -- TMR, voting systems, graceful degradation, watchdog timers.
5. **Networking** -- For Starlink roles: routing protocols, mesh networking, handoff algorithms.
6. **Physics basics** -- Orbital mechanics, signal propagation, atmospheric effects. You don't need a PhD, but understanding the domain helps.
7. **Prepare for many rounds** -- 7-9 rounds is exhausting. Pace yourself and stay energized.
8. **Accept the trade-off** -- Lower cash comp but potentially massive equity upside. Know your risk tolerance.

## Sources

- [SpaceX Software Engineer Interview Questions - InterviewQuery](https://www.interviewquery.com/interview-guides/spacex-software-engineer)
- [SpaceX Interview Process - 4dayweek.io](https://4dayweek.io/interview-process/spacex)
- [Getting a Job Offer at SpaceX - InterviewPal](https://www.interviewpal.com/blog/getting-a-job-offer-at-spacex-interview-process-and-top-questions-to-practice)
- [SpaceX's Interview Process - interviewing.io](https://interviewing.io/spacex-interview-questions)
- [Glassdoor - SpaceX SWE Interview Questions](https://www.glassdoor.com/Interview/SpaceX-Software-Engineer-Interview-Questions-EI_IE40371.0,6_KO7,24.htm)
- [Jointaro - SpaceX Interview Experiences](https://www.jointaro.com/interviews/companies/spacex/)
- r/SpaceX, r/cscareerquestions, Blind (community reports)
