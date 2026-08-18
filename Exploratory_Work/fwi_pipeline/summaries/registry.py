SUMMARY_REGISTRY = {}


def _register():
    from .percentile import PercentileSummary

    SUMMARY_REGISTRY.update({"percentile": PercentileSummary})


_register()
