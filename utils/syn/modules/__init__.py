from .Demosaicing_malvar2004 import demosaicing_CFA_Bayer_Malvar2004
import pyximport; pyximport.install()
from .tone_mapping_cython import CRF_Map_Cython, ICRF_Map_Cython