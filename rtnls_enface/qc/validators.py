from rtnls_enface.base import FundusQCValidator
from rtnls_enface.fundus import Fundus


class GeneralQCValidator(FundusQCValidator):
    """General-purpose fundus QC
    - disc mask must have a single connected component
    - optic disc must be within the image
    - bluriness measures?
    """

    def __call__(self, fundus: Fundus):
        if fundus.disc is None or fundus.bounds is None:
            return False
        if fundus.disc.num_ccs != 1:
            return False
        dist = fundus.disc.distance_to_bounds_2()

        return dist > 5
