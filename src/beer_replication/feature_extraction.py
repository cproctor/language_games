import numpy as np
import arrow

def pairwise(items):
    old = None
    for i in items:
        if old is not None: yield (old, i)
        old = i

def month_count(dt):
    return 12 * dt.year + dt.month

class FeatureExtractor:
    """
    Abstract class for extracting features from chronologically-sorted DataFrame of comments. 
    """
    feature_classes = {}

    def __init__(self, bin_size=5):
        "Override with necessary configuration"
        self.bin_size = bin_size

    def values_to_features(self, values, feature_name):
        "Given a list of values, returns a dict of features, include the argmax"
        features = {"{}_{}".format(feature_name, i) : v for i, v in enumerate(values)}
        features["{}_max".format(feature_name)] = np.argmax(values)
        features["{}_min".format(feature_name)] = np.argmin(values)
        return features

    def extract_features(self, comments):
        "Maps comments to activity-based features."
        assert len(comments) % self.bin_size == 0
        bins = comments.groupby(np.arange(len(comments)) // self.bin_size)
        features = {}
        for feature_class_name, feature_fn in self.feature_classes.items():
            values = [feature_fn(bins.get_group(i)) for i in range(len(bins))]
            feature_class = self.values_to_features(values, feature_class_name)
            features.update(feature_class)
        return features
            
class ActivityFeatureExtractor(FeatureExtractor):
    def __init__(self, bin_size=5, start_month="2007-02"):
        "Override with necessary configuration"
        self.bin_size = bin_size
        self.start_month = arrow.get(start_month)
        self.feature_classes = {
            'freq': self.bin_frequency,
            'month': self.bin_month
        }

    def bin_frequency(self, comments):
        return np.mean([(comments.iloc[i].created_at - comments.iloc[i-1].created_at).days 
                for i in range(1, len(comments))])

    def bin_month(self, comments):
        return month_count(comments.created_at.max()) - month_count(self.start_month)
    
class LinguisticFeatureExtractor(FeatureExtractor):
    "Comments should already have properties 'bigram_score' and 'wv_score'"
    def __init__(self, bin_size=5):
        self.bin_size = bin_size
        self.feature_classes = {
            "bigram": self.bin_bigram_ce,
            "wv": self.bin_word_vector_ll
        }

    def bin_bigram_ce(self, comments):
        return np.mean(comments['bigram_score'])

    def bin_word_vector_ll(self, comments):
        return np.mean(comments['wv_score'])

class InitialModelFeatureExtractor(FeatureExtractor):
    "Comments should already have property 'initial_wv_score'"
    def __init__(self, bin_size=5):
        self.bin_size = bin_size
        self.feature_classes = {
            "initial_wv": self.bin_word_vector_ll
        }

    def bin_word_vector_ll(self, comments):
        return np.mean(comments['initial_wv_score'])






















