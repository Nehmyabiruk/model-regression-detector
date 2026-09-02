FORBIDDEN_ACTIONS = {
    "deploy_model",
    "delete_model",
    "delete_data",
    "change_production_config",
    "rollback_production",
}


def validate_action(
    action: str,
) -> bool:

    normalized = action.strip().lower()

    for forbidden in FORBIDDEN_ACTIONS:

        if forbidden in normalized:
            return False

    return True