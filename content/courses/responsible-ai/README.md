# Responsible AI

Shipping a model means knowing whether you may use the data, what personal data it holds, what the model can do, how it fails different groups, what you let people use it for, and what you owe when something goes wrong. This book turns each of those obligations into an engineering artifact for the system built in [*Build Your Own LLM System*](../build-an-llm/README.md). It is for engineers who want their responsibility work to be as checkable as their code. It is not legal advice, and it is plain about what each check cannot tell you: a lawyer still decides what a license permits.

## What you will be able to do

- Record every training source's license at fetch time and fail a build on anything an allowlist does not permit.
- Write a privacy policy that names every data flow, and measure your PII detector's precision and recall.
- Write a model card and datasheet with intervals on every number and limitations grounded in measurements.
- Gate a release on bias and safety evals, and enforce a usage policy at the gateway that fails closed.
- Review a published claim by the conclusion its readers will draw, and give every open point an owner.

## Ways through this book

Reading every chapter and doing its applied work takes about four weeks at six to eight hours a week.

- **Shipping a model at work** (about two weeks): [Chapter 3: Model and Data Cards](03-model-and-data-cards/README.md), because writing the card exposes the gaps, then [4.1 Bias and safety evals](04-bias-and-safety-evals/01-bias-and-safety-evals.md), then [1.1](01-data-licensing/01-data-licensing-and-the-ledger.md) or [2.1](02-privacy-and-pii/01-privacy-and-pii-policy.md) as the gaps demand.
- **Running an LLM API** (about ten days): [Chapter 5: Usage Policy](05-usage-policy/README.md), then [2.1 Privacy and PII policy](02-privacy-and-pii/01-privacy-and-pii-policy.md), so you decide what you refuse and what you log before your first customer.
- **Reporting a flaw, in someone else's system or yours** (a few evenings): [6.1 Professional responsibility and disclosure](06-professional-responsibility/01-professional-responsibility.md), with [3.1 Datasheet and model card](03-model-and-data-cards/01-datasheet-and-model-card.md) for the card it reviews.
- **The whole book, in order**: Chapters 1 to 6 follow a model from its training data to what you owe after it ships.

## Before you start

The reading needs no prerequisites, but the applied sections assume the system they apply to: the [Corpus Pipeline](../data-engineering/05-corpus-pipeline/README.md) of *Data Engineering* for Chapters 1 and 2, the capstone model of *Build Your Own LLM System* for Chapters 3 and 4, and the [Gateway](../ai-platform-engineering/12-gateway/README.md) of *AI Platform Engineering* for Chapter 5. The evals use confidence intervals from [Probability and Statistics](../mathematics-for-computing/08-probability-statistics/README.md) in *Mathematics for Computing*. Related books are [*Software Craftsmanship*](../software-craftsmanship/README.md), for the threat model a disclosure cites, and [*Systems and Architecture*](../systems/README.md), for postmortems. New to these books? [How to read these books](../../references/how-to-read.md) explains how sections, review questions, and solutions work.
