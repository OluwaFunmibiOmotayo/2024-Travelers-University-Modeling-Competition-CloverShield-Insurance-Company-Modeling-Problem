# 2024-Travelers-University-Modeling-Competition-CloverShield-Insurance-Company-Modeling-Problem
Develop a predictive model to forecast policyholder call frequency (call counts) for CloverShield Insurance, based on customer and policy data.


## Files
train_data.csv - the training set

test_data.csv - the validation set on which you will make your predictions

# Variable Descriptions
ann_prm_amt: Annualized Premium Amount

bi_limit_group: Body injury limit group (SP stands for single split limit coverage, CSL stands for combined single limit coverage)

channel: Distribution channel

newest_veh_age: The age of the newest vehicle insured on a policy (-20 represents non-auto or missing values)

geo_group: Indicates if the policyholder lives in a rural, urban or suburban area

has_prior_carrier: Did the policyholder come from another carrier

home_lot_sq_footage: Square footage of the policyholder’s home lot

household_group: The types of policy in household

household_policy_counts: Number of policies in the household

telematics_ind: Telematic indicator (0 represents auto missing values or didn’t enroll and -2 represents non-auto)

digital_contacts_ind: An indicator to denote if the policy holder has opted into digital communication

12m_call_history: Past one year call count

tenure_at_snapshot: Policy active length in month

pay_type_code: Code indicating the payment method

acq_mthd: The acquisition method (Miss represents missing values)

trm_len_mo: Term length month

pol_edeliv_ind: An indicator for email delivery of documents (-2 represents missing values)

product_sbtyp_grp: Product subtype group

product_sbtyp: Product subtype

call_counts: The number of call count generated by each policy (target variable)
