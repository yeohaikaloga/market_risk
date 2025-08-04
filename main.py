#  from workflow.cotton_basis_workflow import fy24_cotton_basis_workflow
from db.db_connection import get_engine
from position_loader.derivatives_position_loader import DerivativesPositionLoader
#from position_loader.physical_position_loader import PhysicalsPositionLoader
import pandas as pd


if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    prod_engine = get_engine('prod')
   # Load generic curves relative return series


    # Load basis abs return series: fy24_cotton_basis_workflow(write_to_excel=True, apply_smoothing=False)
    # Load physical return series



    uat_engine = get_engine('uat')
    COB_DATE = '2025-08-02'  # KEY HERE 2025-07-29 THIS IS ACTUALLY FOR COB 2025-07-28 so need to insert one day after
    derivatives = DerivativesPositionLoader(date=COB_DATE,
                                            source=uat_engine)  # will need to change to prod_engine later.
    # Load positions df
    product = 'rms'
    deriv_pos_df = derivatives.load_position(date=COB_DATE, product=product)

    print(deriv_pos_df.head())
    print(deriv_pos_df.columns)
    print(deriv_pos_df.derivative_type.unique())
    deriv_pos_df.to_csv(f"position_{product}_{COB_DATE}.csv", index=False)

        #physicals = LoadedPhysicalsPosition(date=COB_DATE, source=uat_engine) # will need to change to prod_engine later.
        #phys_pos_df = physicals.load_position(date=COB_DATE, ors_product='cto') # check ors_product

    # Calculate PnLs --> VaR

    # Format into VaR report
