from discourse_community import DiscourseCommunity
from settings import *

models = [get_month_embedding_filepath(year, 1) for year in [2008, 2017]]
#models = [INITIAL_MODEL] + [get_month_embedding_filepath(year, 1) for year in range(2008, 2018)]
dc = DiscourseCommunity(models)

