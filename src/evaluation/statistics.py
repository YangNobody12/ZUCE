"""
Phase 9C: Statistical Validation & Specialization Metrics
Calculates:
1. Capability Retention: R_t = Score_{student, t} / Score_{teacher, t}
2. Specialization Gain: SG = R_{code} - R_{general} > 0
3. Compression Efficiency: CE = R_{code} / (P_{student} / P_{teacher})
4. Bootstrap 95% Confidence Interval.
"""

import numpy as np
from typing import Dict, List, Any, Tuple

class StatisticalValidator:
    @staticmethod
    def compute_retention_and_specialization(
        teacher_scores: Dict[str, float],
        student_scores: Dict[str, float],
        teacher_params: int,
        student_params: int
    ) -> Dict[str, Any]:
        """
        Computes formal scientific extraction metrics (R_t, SG, CE).
        """
        retention = {}
        for domain, t_score in teacher_scores.items():
            s_score = student_scores.get(domain, 0.0)
            r_t = (s_score / max(t_score, 1e-6)) if t_score > 0 else 0.0
            retention[domain] = round(r_t * 100.0, 2)

        r_code = retention.get("coding", 0.0) / 100.0
        r_gen = retention.get("general", 0.0) / 100.0

        # Specialization Gain SG = R_code - R_general
        specialization_gain = round((r_code - r_gen) * 100.0, 2)

        # Compression Efficiency CE = R_code / (P_student / P_teacher)
        param_ratio = max(student_params / max(teacher_params, 1), 1e-6)
        compression_efficiency = round(r_code / param_ratio, 3)

        is_specialized = specialization_gain > 0.0

        return {
            "retention_percentages": retention,
            "specialization_gain_pct": specialization_gain,
            "compression_efficiency": compression_efficiency,
            "parameter_ratio": round(param_ratio, 3),
            "is_specialist_extraction": is_specialized,
            "scientific_conclusion": (
                f"Specialization Gain: {specialization_gain:+.2f}% | CE: {compression_efficiency}x | "
                f"Status: {'VALID SPECIALIZATION (SG > 0)' if is_specialized else 'GENERIC COMPRESSION'}"
            )
        }

    @staticmethod
    def compute_bootstrap_ci(
        sample_results: List[float],
        n_bootstraps: int = 1000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """Calculates mean and 95% Bootstrap Confidence Interval."""
        if not sample_results:
            return 0.0, 0.0, 0.0

        arr = np.array(sample_results)
        boot_means = []
        n = len(arr)

        for _ in range(n_bootstraps):
            resample = np.random.choice(arr, size=n, replace=True)
            boot_means.append(np.mean(resample))

        mean = float(np.mean(arr))
        alpha = (1.0 - confidence_level) / 2.0
        low = float(np.percentile(boot_means, alpha * 100))
        high = float(np.percentile(boot_means, (1.0 - alpha) * 100))

        return round(mean, 3), round(low, 3), round(high, 3)
