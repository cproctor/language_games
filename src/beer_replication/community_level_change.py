# Exploratory analysis showing change in the community over time
# author: cp

import arrow
from os.path import join
from beer_replication.helpers import *
from settings import *

def get_month_lm_filepath(year, month, ext="arpa"):
    return join(HN_MONTHLY_LM_DIR, HN_MONTHLY_LM_TEMPLATE.format(year, month, ext))

if True: # Generate some 
    months = arrow.Arrow.span_range('month', arrow.get(START_MONTH), arrow.get(END_MONTH))
    labels = [b.format("YYYY-MM") for b, e in months]
    paths = [get_month_lm_filepath(b.year, b.month, 'binary') for b, e in months]
    model = LongitudinalCE(paths, labels)
