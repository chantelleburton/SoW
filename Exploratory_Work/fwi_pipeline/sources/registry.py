SOURCE_REGISTRY = {}


def _register():
    from .era5_present import ERA5PresentSource
    from .era5_historical import ERA5HistoricalSource
    from .hadgem3_historical import HadGEM3HistoricalSource
    from .hadgem3_attribution import HadGEM3AttributionSource

    SOURCE_REGISTRY.update({
        "era5_present": ERA5PresentSource,
        "era5_historical": ERA5HistoricalSource,
        "hadgem3_historical": HadGEM3HistoricalSource,
        "hadgem3_attribution": HadGEM3AttributionSource,
    })


_register()
