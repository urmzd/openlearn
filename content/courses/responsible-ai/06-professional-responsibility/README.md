# Professional Responsibility

Responsibility is the part of engineering that does not compile. Engineers owe duties beyond the employer: to the public, to users, and to the profession. The professional codes write those duties down so they can be cited when they conflict. The ACM Code of Ethics puts the public good first (principle 1.1), asks for honesty about capabilities and limits (1.3), and sets a duty to report risks (2.5, 3.1). The codes, the disclosure process, and the postmortem are tools for making obligations explicit and checkable, so the right action under pressure is the practiced one.

Two practices carry that idea into a running system. Disclosure is a process, not an event: discovery, private report, acknowledgement and validation, remediation, and a coordinated public advisory, with an embargo and timelines agreed in advance. Applied to your own system, it produces a security advisory and a changelog entry consistent with your threat model and release process. And blameless does not mean responsibility-free: a postmortem removes blame from people so the system can be fixed, and still names what the organization owes those affected.

This chapter closes the book with a single section, [Professional responsibility and disclosure](01-professional-responsibility.md), in which you review one consequential claim from the model card: what evidence supports it, what is unknown, who could be harmed if it is overstated, and what wording and product control reduce that risk. You write the review against a design-review checklist and argue the tradeoffs you made, and a complete model answer shows the reasoning behind each paragraph.

> **Learning objectives**
> - Name the duties the ACM and IEEE codes assign beyond the employer, and use them to reason about a conflict.
> - Describe the phases of coordinated vulnerability disclosure and what an advisory for your own system must contain.
> - Review a published claim by the conclusion its audience will draw, and size its evidence.
> - Turn every unresolved point in a review into a note with an owner, a next step, and a done condition.

## Before you begin

This chapter draws on the rest of the book, especially the model card from [Model and Data Cards](../03-model-and-data-cards/README.md), the evals from [Bias and Safety Evals](../04-bias-and-safety-evals/README.md), and the policy from [Usage Policy](../05-usage-policy/README.md). For postmortems and their action items, read [Incident Response and Chaos](../../systems/05-incident-response-and-chaos/) in *Systems and Architecture*. A disclosure cites the threat model of [Threat model (STRIDE on your DFD)](../../software-craftsmanship/08-security/01-threat-model-stride.md) and follows the release process of [Commits, semver, releases (v1.0.0)](../../software-craftsmanship/06-releases-and-deprecation/01-commits-semver-and-releases.md), both in *Software Craftsmanship*. A good warm-up is to read the ACM code with two case studies of your choice, then the chapters of the CERT guide on the disclosure process.

## In this chapter

- [Professional responsibility and disclosure](01-professional-responsibility.md): what responsibility asks of an engineer shipping this system, a design-review checklist, and a fully worked review of the model card's safety claim, from evidence and options to failure modes, open questions, and a decision.

For practice explaining your choices under questioning, see [Defend Your System](../../interview-preparation/defend-your-system/) in *Interview Preparation*. The [Security](../../software-craftsmanship/08-security/) chapter holds the threat model and supply-chain records a disclosure cites.

## Further reading

- [ACM Code of Ethics and Professional Conduct](https://www.acm.org/code-of-ethics) (free).
- CERT, [*Guide to Coordinated Vulnerability Disclosure*](https://certcc.github.io/CERT-Guide-to-CVD/) (free).
- [IEEE Code of Ethics](https://www.ieee.org/about/corporate/governance/p7-8.html) (free).
- ISO/IEC 29147, vulnerability disclosure.
- Leveson and Turner, *An Investigation of the Therac-25 Accidents* (IEEE Computer, 1993).
