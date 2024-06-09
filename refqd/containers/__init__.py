from .extended_me_repertoire import (
    compute_cvt_centroids, ExtendedMapElitesRepertoire, CPUMapElitesRepertoire
)
from .extended_depth_repertoire import ExtendedDepthRepertoire, CPUDepthRepertoire


ExtendedRepertoire = ExtendedMapElitesRepertoire | ExtendedDepthRepertoire


def get_repertoire_type(name: str):
    match name:
        case 'GPU':
            return ExtendedMapElitesRepertoire
        case 'CPU':
            return CPUMapElitesRepertoire
        case 'GPU-Depth':
            return ExtendedDepthRepertoire
        case 'CPU-Depth':
            return CPUDepthRepertoire
        case _:
            raise NotImplementedError(name)


__all__ = [
    'compute_cvt_centroids', 'ExtendedMapElitesRepertoire', 'CPUMapElitesRepertoire',
    'ExtendedDepthRepertoire', 'CPUDepthRepertoire',
    'ExtendedRepertoire',
    'get_repertoire_type',
]
