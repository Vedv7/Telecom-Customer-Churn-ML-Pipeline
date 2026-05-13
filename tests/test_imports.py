"""Smoke test: package imports resolve (CI runs without local Excel data)."""


def test_import_telecom_churn():
    import telecom_churn  # noqa: F401
    from telecom_churn import train as train_mod  # noqa: F401
