from cores.state import State

MAX_RETRIES = 3


def node_aggregate(state: State):

    errors = state.get("errors", [])
    retry_counts = state.get("retry_counts", {}).copy()
    processed_error_count = state.get("processed_error_count", 0)

    # Only process errors that were added since the last aggregate
    new_errors = errors[processed_error_count:]

    retry_targets = set()

    for error in new_errors:

        # This error should not be retried
        if not error.retryable:
            continue

        # No specific tool to retry
        if error.tool is None:
            continue

        tool = error.tool

        # Number of previous retries for this tool
        current_count = retry_counts.get(tool, 0)

        # Retry if the limit has not been reached
        if current_count < MAX_RETRIES:
            retry_counts[tool] = current_count + 1
            retry_targets.add(tool)

    return {
        "retry_counts": retry_counts,
        "retry_targets": list(retry_targets),
        # Mark all current errors as processed
        "processed_error_count": len(errors),
    }


def route_after_aggregate(state: State):
    retry_list = []

    retry_targets = state.get("retry_targets", [])
    if not retry_targets:
        retry_list = ["summary"]

    for target in retry_targets:
        if target == "document_qa":
            retry_list.append("document_qa")

        elif target == "local_knowledge":
            retry_list.append("local_knowledge")

        else:
            retry_list.append("web_search")

    return retry_list
