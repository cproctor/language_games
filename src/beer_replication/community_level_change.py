# Exploratory analysis showing change in the community over time
# author: cp

import arrow
from os.path import join
from helpers import *
from settings import *

if True: # Generate some 
    months = arrow.Arrow.span_range('month', arrow.get(START_MONTH), arrow.get(END_MONTH))
    labels = [b.format("YYYY-MM") for b, e in months]
    paths = [get_month_lm_filepath(b.year, b.month, 'binary') for b, e in months]
    model = LongitudinalCE(paths, labels)
