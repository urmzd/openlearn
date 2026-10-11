# Professional Responsibility

## Overview

- **Primary references**: [ACM Code of Ethics and Professional Conduct](https://www.acm.org/code-of-ethics) (free); CERT, [*Guide to Coordinated Vulnerability Disclosure*](https://certcc.github.io/CERT-Guide-to-CVD/) (free)
- **Supplementary**: [IEEE Code of Ethics](https://www.ieee.org/about/corporate/governance/p7-8.html) (free); ISO/IEC 29147 (vulnerability disclosure); Leveson and Turner, *An Investigation of the Therac-25 Accidents* (IEEE Computer, 1993)
- **Prerequisites**: the rest of this track; [Incident Response and Chaos](../../systems/05-incident-response-and-chaos/) for postmortems
- **Estimated time**: 3 to 4 h in course Pass 11

## Key Takeaways

- **Engineers owe duties beyond the employer**: to the public, to users, and to the profession. The codes write those duties down so they can be cited when they conflict.
- **Disclosure is a process, not an event**: private report, acknowledgement, fix, coordinated public advisory, with timelines agreed in advance.
- **Blameless does not mean responsibility-free**: a postmortem removes blame from people so the system can be fixed, and still names what the organization owes those affected.

## How to Study

Read the ACM code with two case studies of your choice, then the CERT guide chapters on the disclosure process. In the course, `ethics.06` is a proof module graded by rubric: you write a disclosure for a planted flaw in your own system and argue the trade-offs you made.

---

# Concepts & Techniques

## Core Insight

Responsibility is the part of engineering that does not compile. The codes, the disclosure process, and the postmortem are tools for making obligations explicit and checkable, so the right action under pressure is the practiced one.

## 1. Codes and conflicts

**Key ideas**:
- **The public good first** (ACM 1.1), honesty about capabilities and limits (1.3), and the duty to report risks (2.5, 3.1).

## 2. Coordinated disclosure

**Key ideas**:
- **Phases**: discovery, report, validation, remediation, public disclosure, with an embargo and a deadline.
- **Applied**: a security advisory and changelog entry for your own system, consistent with your threat model (`craft.17`) and release process (`craft.11`).

## Course modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `ethics.06` | Professional responsibility and disclosure | proof | 11 |

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `ethics.06` | [Professional responsibility and disclosure](01-professional-responsibility.md) | proof | 11 |
<!-- /ol:chapters -->

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Responsible AI](../) | the track overview and how the six topics connect |
| [Security](../../software-craftsmanship/11-security/) | the threat model and supply-chain records a disclosure cites |
| [Incident Response and Chaos](../../systems/05-incident-response-and-chaos/) | postmortems and their action items |
| [Defend Your System](../../interviews/defend-your-system/) | explaining your choices under questioning |
