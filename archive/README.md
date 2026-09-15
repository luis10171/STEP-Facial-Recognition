# Original recovered source

`original_main.py` is an unchanged copy of the script recovered at the start of restoration. Its SHA-256 is:

```text
2294a2a49a8727869562a7538f467fb1aecc4f16a95ec2f21f7226f19fee6ea4
```

It documents the original science-fair implementation. It is not supported by the restored requirements and should not be used as the application entry point. It depends on absent pickle databases and imports/opens devices at module scope. Untrusted pickle files can execute code when loaded. Use the top-level `main.py` instead.
