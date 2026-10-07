"""Open-vocabulary detection components required by inference."""

from .tgfa_ovdfine_decoder import TGFAOVDFINETransformer
from .tgfa_ovdfine_decoder_obb import TGFAOVOBBDFINETransformer

__all__ = ["TGFAOVDFINETransformer", "TGFAOVOBBDFINETransformer"]
