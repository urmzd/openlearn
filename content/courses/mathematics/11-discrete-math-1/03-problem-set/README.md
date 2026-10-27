# Discrete math 1 problem set: sets, counting, proofs, induction, relations, bijections

The chapters ahead are arguments as much as code. [Graphs, DAGs, and an iterative topological sort](../../12-discrete-math-2/01-graphs-dags-and-topological-sort.md) must order every node of the autograd graph before backward runs, and the only convincing evidence that it does is an induction over the traversal. [Modular arithmetic, hashing, and PCG32](../../12-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) builds a generator whose state wraps modulo $2^{64}$ and whose output is a counted rotation of bits. [Counting: parameters, FLOPs, KV bytes, and memory plans](../01-counting-params-flops-kv-bytes.md) turns a `ModelConfig` into counts that decide whether a model fits on a laptop, and [Injective, surjective, bijective: the GPT-2 byte map](../02-injective-surjective-bijective-gpt2-byte-map.md) proves a byte table invertible. In *AI Engineering*, [Masks: causal, padding, sliding window, additive](../../../ai-engineering/11-transformer/02-masks.md) builds attention masks from boolean algebra, and [Softmax in C: three-pass and online two-pass](../../../ai-engineering/22-kernels/04-softmax-three-pass-and-online.md) streams a softmax in one pass under a loop invariant.

This set gives the vocabulary and the proof habits those sections assume: sets and logic, counting, the standard proof methods, induction, relations, and functions. It needs only high-school algebra and the overview in the [chapter opening](../README.md), and it is best worked before the two sections of this chapter.

The set has 23 problems with exact answers (34 parts) and 18 proofs, whose model proofs are also in [`course/solutions/solve/S-M05/`](../../../../../course/solutions/solve/S-M05). Every problem is followed by a collapsed worked answer with the final result or a complete model proof, and the common wrong answers and why they are wrong. Try each problem before opening its answer.

> **Learning objectives**
> - Manipulate sets and quantified statements, including negating nested quantifiers, and read an attention mask as a set.
> - Count with the product, permutation, combination, and stars-and-bars rules, and apply them to parameters, FLOPs, and KV-cache bytes.
> - Write complete direct, contrapositive, contradiction, case, and pigeonhole proofs.
> - Prove statements by induction and prove loops correct with invariants.
> - Classify relations, and prove functions injective, surjective, or bijective.

## In this part

1. [Tools and a worked proof](01-tools-and-worked-proof.md): Connect set, logic, counting, and induction tools to a worked argument.
2. [Sets, logic, and counting](02-sets-logic-and-counting.md): Work the set, quantified-logic, and counting problems before revealing solutions.
3. [Proofs, induction, and invariants](03-proofs-induction-and-invariants.md): Choose a proof method and connect induction to loop invariants.
4. [Relations, functions, and validation](04-relations-functions-and-validation.md): Check relation properties and prove function/bijection claims with explicit domains.
