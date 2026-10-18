# Story LSTM reference calibration

The full `MS-L3/lstm-ts` and `MS-L3/lstm-ts-eval` rows were measured again on
2026-10-18 using seeds 0 through 4 on native GitHub-hosted Linux x86_64,
CPython 3.11.17, NumPy 2.4.6, and OpenBLAS 0.3.31.188.0 with one BLAS thread. The runs use the learner's
`course/ref/entry/python` project rather than the harness dependency environment.
[The measurement run](https://github.com/urmzd/openlearn/actions/runs/39090297858)
retains each seed's training and held-out evaluation JSON.

The reference, fixture, and training parameters were unchanged: LSTM,
32-dimensional embeddings, hidden size 128, BPTT 32, batch 32, learning rate
0.01, gradient clip 1.0, and 1,500 steps on `MS-L2/train.bin`. The raw values
are committed in `course/fixtures/ref-thresholds.tsv`. The normal threshold
calculation, mean plus three sample standard deviations over five seeds,
gives 0.4889299547 training validation bits per character and 0.4614008719
held-out bits per byte. The separate requirement to beat the NPLM at 0.6261
bits per byte remains unchanged.

The stored calibration had a narrower spread than these native Linux
measurements. Reference training on nightly produced 0.452339 validation
bits per character, outside the old 0.4499187805 bound but within the newly
measured reference distribution. Floating-point training results can differ
across NumPy and BLAS runtimes, so record measurements on the hosted runtime
rather than choosing a threshold by hand.

To repeat calibration from an assembled reference learner:

```sh
ol milestone MS-L3 --ref-deps --step lstm-ts --step lstm-ts-eval \
  --record-thresholds --seeds 5
```
