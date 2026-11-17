def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "e2e: end-to-end tests — require ANTHROPIC_API_KEY and RUN_E2E=1",
    )
