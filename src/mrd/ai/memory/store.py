from mrd.ai.memory.models import RegressionIncident


class IncidentMemory:
    """
    In-memory incident store.

    This is the first implementation of the memory
    interface. It can later be replaced by PostgreSQL,
    pgvector, or another persistent backend.
    """

    def __init__(self) -> None:
        self._incidents: list[
            RegressionIncident
        ] = []

    def save(
        self,
        incident: RegressionIncident,
    ) -> None:
        """Store a regression incident."""

        self._incidents.append(
            incident
        )

    def get_all(
        self,
    ) -> list[RegressionIncident]:
        """Return all stored incidents."""

        return list(
            self._incidents
        )

    def get_by_model(
        self,
        model_name: str,
    ) -> list[RegressionIncident]:
        """Return incidents for one model."""

        return [
            incident
            for incident in self._incidents
            if incident.model_name == model_name
        ]