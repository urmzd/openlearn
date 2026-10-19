"""Forward and reverse KL distillation objectives for L12.4."""
from __future__ import annotations
import numpy as np


def _logsoftmax(x, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    y = np.asarray(x, dtype=np.float64) / temperature
    m = np.max(y, axis=-1, keepdims=True)
    return y - m - np.log(np.exp(y - m).sum(axis=-1, keepdims=True))


def forward_kl(student_logits, teacher_logits, temperature=1.0):
    log_s, log_t = _logsoftmax(student_logits, temperature), _logsoftmax(teacher_logits, temperature)
    if log_s.shape != log_t.shape:
        raise ValueError("student and teacher shapes differ")
    p = np.exp(log_t)
    return float(np.mean(np.sum(p * (log_t - log_s), axis=-1)) * temperature**2)


def reverse_kl(student_logits, teacher_logits, temperature=1.0):
    log_s, log_t = _logsoftmax(student_logits, temperature), _logsoftmax(teacher_logits, temperature)
    if log_s.shape != log_t.shape:
        raise ValueError("student and teacher shapes differ")
    q = np.exp(log_s)
    return float(np.mean(np.sum(q * (log_s - log_t), axis=-1)) * temperature**2)
