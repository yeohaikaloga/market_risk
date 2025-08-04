#  from workflow.cotton_basis_workflow import fy24_cotton_basis_workflow
from db.db_connection import get_engine
from position.loaded_derivatives_position import LoadedDerivativesPosition
#from position.loaded_physical_position import LoadedPhysicalsPosition
import pandas as pd


if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    prod_engine = get_engine('prod')
   # Load generic curves relative return series


    # Load basis abs return series: fy24_cotton_basis_workflow(write_to_excel=True, apply_smoothing=False)
    # Load physical return series



    uat_engine = get_engine('uat')
    COB_DATE = '2025-07-29'  # BUT THIS IS ACTUALLY FOR COB 2025-07-28 so need to insert one day after

    # Load positions df
    product = 'cotton'
    if product == 'cotton':
        derivatives = LoadedDerivativesPosition(date=COB_DATE, source=uat_engine)  # will need to change to prod_engine later.
        deriv_pos_df = derivatives.load_position(date=COB_DATE, opera_product='cto')
        print(deriv_pos_df.head())
        print(deriv_pos_df.columns)
        print(deriv_pos_df.derivative_type.unique())

        #physicals = LoadedPhysicalsPosition(date=COB_DATE, source=uat_engine) # will need to change to prod_engine later.
        #phys_pos_df = physicals.load_position(date=COB_DATE, ors_product='cto') # check ors_product

    # Calculate PnLs --> VaR

    # Format into VaR report
