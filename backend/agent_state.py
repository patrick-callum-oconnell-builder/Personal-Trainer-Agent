    def _validate_status(self, status):
        """
        Validate that status is one of the allowed values.
        """
        try:
            from backend.agent_orchestration.state_handler import AgentState as StateHandlerAgentState
            enum_statuses = [e.value for e in StateHandlerAgentState]
        except Exception:
            enum_statuses = []
        valid_statuses = ["active", "awaiting_user", "awaiting_tool", "error", "done"] + enum_statuses
        if status not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}") 