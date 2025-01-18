import numpy as np
from math import gcd
from fractions import Fraction
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit.circuit.library import QFT
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class RunResult:
    a: int
    power: int
    precision: int
    probability: float  # best single measurement probability
    total_valid_probability: float  # sum of all valid measurement probabilities
    phase: float
    period: Optional[int]
    factors: Tuple[Optional[int], Optional[int]]

class AdaptiveShor:
    def __init__(self, service_token: str, max_runs: int = 5):
        self.service = QiskitRuntimeService(
            channel="ibm_quantum",
            token=service_token
        )
        
        # Configuration
        self.TEST_VALUES = [2, 3, 5, 7, 11, 13, 17, 19]
        self.MAX_RUNS = max_runs
        self.TOP_A_VALUES = 3
        self.SHOTS = 500
        
        # Statistics tracking
        self.a_value_history = defaultdict(list)

    def _modular_exponentiation(self, a: int, x: int, N: int) -> int:
        result = 1
        a = a % N
        while x > 0:
            if x & 1:
                result = (result * a) % N
            a = (a * a) % N
            x = x >> 1
        return result

    def _create_qpe_circuit(self, a: int, N: int, precision: int, power: int) -> QuantumCircuit:
        qc = QuantumCircuit(precision + 1, precision)
        
        # Initialize control in superposition
        for i in range(precision):
            qc.h(i)
            
        # Initialize target to |1⟩
        qc.x(precision)
        
        # Apply controlled U operations
        for i in range(precision):
            angle = (2 * np.pi * self._modular_exponentiation(a, power * 2**i, N)) / N
            qc.cp(angle, i, precision)
        
        # Apply inverse QFT
        qc.append(QFT(precision).inverse(), range(precision))
        qc.measure(range(precision), range(precision))
        
        return qc

    def _check_factors(self, a: int, period: int, N: int) -> Tuple[Optional[int], Optional[int]]:
        if period % 2 != 0:
            return None, None
            
        x = pow(a, period//2, N)
        if x in [0, N-1]:
            return None, None
            
        factor1 = gcd(x - 1, N)
        factor2 = gcd(x + 1, N)
        
        if factor1 in [1, N] or factor2 in [1, N]:
            return None, None
            
        return min(factor1, factor2), max(factor1, factor2)

    def _run_iteration(self, N: int, run_number: int) -> Optional[Tuple[int, int]]:
        if run_number == 0:
            test_values = [a for a in self.TEST_VALUES if gcd(a, N) == 1]
        else:
            # Use the top 3 a values that gave highest total valid probability
            a_probabilities = {}
            for a, results in self.a_value_history.items():
                avg_prob = np.mean([r.total_valid_probability for r in results])
                a_probabilities[a] = avg_prob
            test_values = sorted(a_probabilities.items(), key=lambda x: x[1], reverse=True)[:self.TOP_A_VALUES]
            test_values = [a for a, _ in test_values]

        print(f"\nRun {run_number + 1}: Testing a values {test_values}")
        
        results_this_run = []
        total_prob_this_run = 0.0  # Track total probability across all a values
        precision = len(bin(N)[2:]) + 3  # Base precision n + 3 where n is number of bits in N
        
        for a in test_values:
            print(f"Testing a={a} with precision={precision}, shots={self.SHOTS}")
            
            backend = self.service.least_busy(simulator=False)
            qc = self._create_qpe_circuit(a, N, precision, power=1)
            transpiled_qc = transpile(qc, backend=backend)
            
            sampler = Sampler(mode=backend)
            result = sampler.run([transpiled_qc], shots=self.SHOTS).result()[0]
            
            counts = result.data.c.get_counts()
            max_probability = 0
            total_valid_probability = 0  # Track total probability of valid measurements
            best_factors = (None, None)
            best_period = None
            best_phase = None
            
            for bitstring, count in counts.items():
                phase = int(bitstring, 2) / (2**precision)
                probability = count / self.SHOTS
                
                frac = Fraction(phase).limit_denominator(N)
                period = frac.denominator
                
                factors = self._check_factors(a, period, N)
                if factors[0]:
                    total_valid_probability += probability
                    if probability > max_probability:
                        max_probability = probability
                        best_factors = factors
                        best_period = period
                        best_phase = phase
            
            total_prob_this_run += total_valid_probability
            
            run_result = RunResult(
                a=a,
                power=1,
                precision=precision,
                probability=max_probability,
                total_valid_probability=total_valid_probability,
                phase=best_phase if best_phase is not None else phase,
                period=best_period,
                factors=best_factors
            )
            
            results_this_run.append(run_result)
            self.a_value_history[a].append(run_result)
            
            if best_factors[0] and total_valid_probability > 0.15:
                print(
                    f"Found factors with a={a}:"
                    f"\n  Phase: {best_phase:.6f}"
                    f"\n  Period: {best_period}"
                    f"\n  Factors: {best_factors[0]}, {best_factors[1]}"
                    f"\n  Best single measurement probability: {max_probability:.4f}"
                    f"\n  Total probability of valid measurements: {total_valid_probability:.4f}"
                )
                return best_factors

        print(f"Run {run_number + 1} Summary:")
        print(f"Total probability of valid factors across all measurements: {total_prob_this_run:.4f}")
        
        # Return best result from this run if it meets probability threshold
        valid_results = [r for r in results_this_run if r.factors[0] and r.total_valid_probability > 0.15]
        if valid_results:
            best_result = max(valid_results, key=lambda r: r.total_valid_probability)
            return best_result.factors
            
        return None

    def factor(self, N: int) -> Optional[Tuple[int, int]]:
        if N % 2 == 0:
            return 2, N//2
        if pow(N, 0.5).is_integer():
            sqrt_n = int(pow(N, 0.5))
            return sqrt_n, sqrt_n

        self.a_value_history.clear()

        for run in range(self.MAX_RUNS):
            factors = self._run_iteration(N, run)
            if factors:
                print(f"\nFactors found in run {run + 1}: {factors[0]} × {factors[1]}")
                
                print("\nPerformance Analysis:")
                for a in self.a_value_history:
                    avg_valid_prob = np.mean([r.total_valid_probability for r in self.a_value_history[a]])
                    print(f"a={a}: Avg total valid probability={avg_valid_prob:.4f}")
                    
                return factors

        print(f"\nFailed to factor {N} after {self.MAX_RUNS} runs")
        return None
    
if __name__ == "__main__":
    token = "Token"
        
    shor = AdaptiveShor(token, max_runs=5)
    N = 299
    print(f"\nFactoring {N}...")
    factors = shor.factor(N)
    
    if factors:
        print(f"Found: {N} = {factors[0]} × {factors[1]}")
    else:
        print(f"Failed to factor {N}")