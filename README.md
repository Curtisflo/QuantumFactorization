# Semi-Classical Shor's Implementation

A bare-bones implementation of Shor's algorithm using semi-classical quantum phase estimation. This approach trades off quantum resource requirements for classical post-processing by iteratively selecting coprime values to maximize factoring probability.

## Implementation Notes

QPE circuit construction uses minimal qubits - just enough for phase estimation plus a single target qubit. Phase estimation precision is set to `log2(N) + 3` qubits. The inverse QFT is applied without approximation.

Each QPE run tests a value 'a' coprime to N, looking for order-finding results that yield non-trivial factors. Key optimizations:

- Initial 'a' values are drawn from small primes to minimize circuit depth
- Subsequent iterations prioritize 'a' values that produced the highest total valid probability in previous runs
- Success threshold of 15% total valid probability filters out noisy results

The full operator U = U^(2^k) is decomposed into controlled phase rotations using modular exponentiation, rather than constructing the full controlled multiplication circuit.

## Usage

```python
shor = AdaptiveShor("ibm-token")
factors = shor.factor(N)  # Returns tuple of factors if found
```

Sample output format:
```
Run 1: Testing a values [2, 3, 5, 7, 11, 17, 19]
Testing a=2 with precision=12, shots=500
...
Found factors with a=11:
  Phase: 0.334732
  Period: 6
  Factors: 13, 23
  Total probability: 0.172
```

## Design Choices

- Fixed 500 shots per circuit execution - balances between statistics and queue time
- Maximum 5 iterations before giving up
- Top 3 'a' values carried forward between iterations
- Phase measurements converted to fractions using continuous fraction expansion with limit denominator N
- Modular exponentiation handled classically

## Dependencies

- Qiskit
- Qiskit IBM Runtime
- NumPy

## Known Limitations

- Not optimized for transpilation to specific hardware topologies
- No error mitigation strategies implemented
- Period finding becomes unreliable above ~11 bit numbers on current hardware
