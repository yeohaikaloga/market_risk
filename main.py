from workflow.cotton_basis_workflow import fy24_cotton_basis_workflow
from db.db_connection import get_engine
from position.loaded_derivatives_position import LoadedDerivativesPosition

if __name__ == '__main__':

    #fy24_cotton_basis_workflow(write_to_excel=True, apply_smoothing=False)

    uat_engine = get_engine('uat')
    COB_DATE = '2025-07-29' # BUT THIS IS ACTUALLY FOR COB 2025-07-28 so need to insert one day after
    derivatives = LoadedDerivativesPosition(date=COB_DATE, source=uat_engine) #will need to change to prod_engine later.
    derivatives_positions = derivatives.load_position(date=COB_DATE, opera_product='cto', source=uat_engine, )
    print(derivatives_positions.head())

