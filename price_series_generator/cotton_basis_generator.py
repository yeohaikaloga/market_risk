from price_series_generator.generated_price_series import PriceSeriesGenerator
from utils.date_utils import get_weekdays_between_list
from utils.date_utils import get_prev_biz_days_list
import numpy as np
import pandas as pd
import math

crop_dict = {'Brazilian': {'grade': 'M_1-1/8"_std', 'type': None},
             'Burkina Faso Bola/s': {'grade': 'SM_1-1/8"_h', 'type': 'Bola/s'},
             'Ivory Coast Manbo/s': {'grade': 'SM_1-1/8"_h', 'type': 'Manbo/s'},
             'Mali Juli/s': {'grade': 'SM_1-1/8"_h', 'type': 'Juli/s'},
             'Memphis/Orleans/Texas': {'grade': 'M_1-1/8"_std', 'type': 'MOT'},
             'A Index': {'grade': 'M_1-1/8"_std', 'type': 'A Index'}}


class CottonBasisGenerator(PriceSeriesGenerator):

    def __init__(self, futures_price_df, contract_expiry_dates, contract_roll_dates,
                 cob_date, window, trailing_days_before_start, source):
        self.futures_price_df = futures_price_df
        self.contract_expiry_dates = contract_expiry_dates
        self.contract_roll_dates = contract_roll_dates
        self.cob_date = cob_date
        self.window = window
        self.trailing_days_before_start = trailing_days_before_start
        self.source = source
        self.start_date = get_prev_biz_days_list(date=cob_date, no_of_days=window + trailing_days_before_start)[0]

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

    def generate_crop_basis(self, physical_contract, crop_price_df_by_year):

        start_date = self.start_date
        cob_date = self.cob_date
        crop_name = physical_contract.instrument_id

        print(f"\nProcessing {crop_name}...")

        # Step 3: Basis logic: cotlook switch + contract_ref_loader switch + smoothing gic
        valid_basis_columns = []
        basis_index = pd.to_datetime(get_weekdays_between_list(start_date, cob_date))
        crop_reindexed = crop_price_df_by_year.reindex(basis_index)
        futures_reindexed = self.futures_price_df.reindex(basis_index)
        crop_basis_df = pd.concat([crop_reindexed, futures_reindexed], axis=1)
        crop_basis_df.index.name = 'tdate'
        print(crop_basis_df.head())
        print(crop_basis_df.tail())

        crop_year_ar_switch_dates = []

        for contract in self.futures_price_df.columns:
            contract_expiry = self.contract_expiry_dates[contract]
            for crop_year in crop_price_df_by_year.columns:
                if physical_contract.crop_year_type == 'cross':
                    ref_year, ref_next_year = map(int, crop_year.split("/"))
                else:
                    ref_year = int(crop_year)
                    ref_next_year = ref_year + 1
                crop_year_start = pd.Timestamp(f"{ref_year}-08-01")
                crop_year_end = pd.Timestamp(f"{ref_next_year}-07-31")
                crop_year_ar_switch = pd.Timestamp(f"{ref_year}-07-31")
                # next crop year's abs ret starts from last biz day in Jul (based on next crop year's price in first
                # biz day in Aug)
                crop_year_ar_switch_dates.append(crop_year_ar_switch)

                # This is to allow Z contract_ref_loader to be taken against current and next crop year
                contract_year = 2020 + int(contract[-1])
                is_ctz_exception = (contract[2] == 'Z' and (contract_year == ref_next_year or
                                                            contract_year == ref_year))

                if crop_year_start <= contract_expiry <= crop_year_end or is_ctz_exception:
                    col_name = f"{crop_year.replace('20', '')} vs {contract}"
                    crop_basis_df[col_name] = crop_basis_df[crop_year].shift(-1) - crop_basis_df[contract]
                    crop_basis_df[col_name] = crop_basis_df[col_name].interpolate(method='linear',
                                                                                  limit_area='inside')
                    crop_basis_df[col_name + ' (sm)'] = \
                        CottonBasisGenerator.smooth_basis(crop_basis_df[col_name], crop_name)[0]
                    crop_basis_df[col_name + ' (sm) w'] = \
                        CottonBasisGenerator.smooth_basis(crop_basis_df[col_name], crop_name)[1]
                    valid_basis_columns.append(col_name)

        for col_name in valid_basis_columns:
            crop_basis_df[col_name + ' AR'] = crop_basis_df[col_name] - crop_basis_df[col_name].shift(1)
            crop_basis_df[col_name + ' AR (sm)'] = crop_basis_df[col_name + ' (sm)'] - crop_basis_df[
                col_name + ' (sm)'].shift(1)

        print(crop_basis_df.head())
        print(basis_index)

        sorted_all_switch_dates = (sorted(
                    dates for dates in set(self.contract_roll_dates.values()).union(crop_year_ar_switch_dates)
                    if pd.Timestamp(start_date) <= dates <= pd.Timestamp(self.cob_date))
                                   + [pd.Timestamp(self.cob_date)])
        print(sorted_all_switch_dates)

        abs_ret_conditions = ([basis_index < sorted_all_switch_dates[0]] +
                              [((basis_index >= sorted_all_switch_dates[i]) &
                                (basis_index < sorted_all_switch_dates[i + 1]))
                               for i in range(len(sorted_all_switch_dates) - 1)])
        abs_ret_cols = [col for col in crop_basis_df.columns if col.endswith(' AR')]
        smooth_abs_ret_cols = [col for col in crop_basis_df.columns if col.endswith(' AR (sm)')]
        abs_ret_choices = [crop_basis_df[col] for col in abs_ret_cols][:len(abs_ret_conditions)]
        smooth_abs_ret_choices = [crop_basis_df[col] for col in smooth_abs_ret_cols][:len(abs_ret_conditions)]
        crop_basis_df['final AR series'] = np.select(abs_ret_conditions, abs_ret_choices, default=0)
        crop_basis_df['final AR (sm) series'] = np.select(abs_ret_conditions, smooth_abs_ret_choices, default=0)
        return crop_basis_df

    def generate_all_crop_basis_return_series(self, physical_contracts_and_prices: list[tuple],
                                              apply_smoothing: bool = True):
        basis_df = pd.DataFrame()

        for physical_contract, crop_price_df_by_year in physical_contracts_and_prices:
            crop_name = physical_contract.instrument_id
            crop_basis_df = self.generate_crop_basis(physical_contract=physical_contract,
                                                     crop_price_df_by_year=crop_price_df_by_year)
            print(f"Generated basis for {crop_name}")

            # Extract 'final AR (sm) series' and rename column
            col_name = 'final AR (sm) series' if apply_smoothing else 'final AR series'
            summary_cols = crop_basis_df[[col_name]].copy()
            summary_cols.columns = [f"{crop_name} {col_name}" for col_name in summary_cols.columns]

            # Combine into basis_df
            basis_df = pd.concat([basis_df, summary_cols], axis=1)

        return basis_df

