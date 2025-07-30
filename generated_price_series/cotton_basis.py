from generated_price_series.generated_price_series import PriceSeriesGenerator
import numpy as np
import pandas as pd
import math


class CottonBasisGenerator(PriceSeriesGenerator):

    @staticmethod
    def smooth_basis(price_series, origin):

        origin_multiples = {'A Index': 0.05, 'Memphis/Orleans/Texas': 0.25, 'Brazilian': 0.25,
                            'Ivory Coast Manbo/s': 0.25, 'Burkina Faso Bola/s': 0.25, 'Mali Juli/s': 0.25}

        multiple = origin_multiples.get(origin)
        if multiple is None:
            raise ValueError(f"Unsupported origin: {origin}. Only Cotlook or specified origins allowed.")

        original_price_series = price_series.dropna()
        revised_price_series = original_price_series.copy()
        annotation_series = pd.Series(index=original_price_series.index, dtype=object)

        # Initialize
        ref_idx = original_price_series.index[0]
        reference_price = original_price_series.loc[ref_idx]
        revised_price_series.loc[ref_idx] = round(round(reference_price, 2) / multiple) * multiple
        annotation_series.loc[ref_idx] = 'start'

        for i in range(1, len(original_price_series)):
            idx = original_price_series.index[i]
            curr_price = round(original_price_series.loc[idx], 2)
            jump = round(curr_price - reference_price, 2)

            if 0 < abs(jump) < multiple:
                # Spurious jump — set to NaN
                revised_price_series.loc[idx] = np.nan
                annotation_series.loc[idx] = f's, ref: {round(reference_price, 2)}'
            else:
                if jump == 0:
                    # No movement
                    revised_price_series.loc[idx] = revised_price_series.loc[original_price_series.index[i - 1]]
                    annotation_series.loc[idx] = f'ns nm eq ref: {round(reference_price, 2)}'
                elif jump > 0:
                    # Up move — round down
                    revised_price_series.loc[idx] = math.floor(curr_price / multiple + 0.001) * multiple
                    annotation_series.loc[idx] = f'ns um rd ref: {round(reference_price, 2)}'
                else:
                    # Down move — round up
                    revised_price_series.loc[idx] = math.ceil(curr_price / multiple - 0.001) * multiple
                    annotation_series.loc[idx] = f'ns dm ru ref: {round(reference_price, 2)}'

                reference_price = curr_price

        # Interpolate spurious (NaN) values
        revised_price_series = revised_price_series.interpolate(method='linear', limit_area='inside').ffill()

        return revised_price_series, annotation_series
