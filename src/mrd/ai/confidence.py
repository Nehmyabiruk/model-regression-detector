def calculate_overall_confidence(
    evidence_strength: float,
    historical_support: float,
    agreement: float,
) -> float:

    values = [
        evidence_strength,
        historical_support,
        agreement,
    ]

    return sum(values) / len(values)