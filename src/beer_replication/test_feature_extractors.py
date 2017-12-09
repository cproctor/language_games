from settings import *
from generate import *
import pandas as pd

comments = pd.read_csv(get_month_filepath(2010, 1), parse_dates=['created_at'])
c = comments.sample(20).sort_values(by=['created_at'])

afe = ActivityFeatureExtractor()
f = afe.extract_features(c)

