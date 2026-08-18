METRIC_REGISTRY = {}


def _register():
    from .fwi import FWIMetric

    METRIC_REGISTRY.update({"fwi": FWIMetric})


_register()
