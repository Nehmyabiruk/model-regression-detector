from mrd.ai.memory.models import RegressionIncident


def calculate_similarity(
    current: dict,
    historical: dict,
) -> float:
    """
    Calculate a simple evidence similarity score.

    The score is based on shared regression signals.
    """

    score = 0.0
    total = 0.0

    current_performance = current.get(
        "performance",
        {},
    )

    historical_performance = historical.get(
        "performance",
        {},
    )

    if current_performance and historical_performance:
        total += 1.0

        if (
            set(current_performance.keys())
            & set(historical_performance.keys())
        ):
            score += 1.0

    current_drift = {
        item.get("feature")
        for item in current.get(
            "drift",
            [],
        )
    }

    historical_drift = {
        item.get("feature")
        for item in historical.get(
            "drift",
            [],
        )
    }

    if current_drift or historical_drift:
        total += 1.0

        intersection = (
            current_drift
            & historical_drift
        )

        union = (
            current_drift
            | historical_drift
        )

        if union:
            score += (
                len(intersection)
                / len(union)
            )

    if total == 0:
        return 0.0

    return score / total


def find_similar_incidents(
    current_evidence: dict,
    incidents: list[RegressionIncident],
    minimum_similarity: float = 0.5,
) -> list[tuple[RegressionIncident, float]]:
    """
    Find historical incidents similar to the current
    regression.
    """

    matches = []

    for incident in incidents:

        similarity = calculate_similarity(
            current=current_evidence,
            historical=incident.evidence,
        )

        if similarity >= minimum_similarity:
            matches.append(
                (
                    incident,
                    similarity,
                )
            )

    matches.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return matches