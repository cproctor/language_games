

def add_min(data, feature_class_name):
    "Given a feature name, adds a new feature indicating the index of the minimum value for that feature"
    features = ['{}_{}'.format(feature_class_name, i) for i in ['0', '1', '2', '3']]
    return data.assign(**{feature_class_name + '_min': np.argmin(data[features].values, axis=1)})

def add_max(data, feature_class_name):
    "Given a feature name, adds a new feature indicating the index of the maximum value for that feature"
    features = ['{}_{}'.format(feature_class_name, i) for i in ['0', '1', '2', '3']]
    return data.assign(**{feature_class_name + '_max': np.argmax(data[features].values, axis=1)})

def feature_class(name, use_max=True, use_min=False):
    "Generates a list of strings for four buckets of features, optionally including min and max"
    suffixes = ['0', '1', '2', '3']
    if use_min: suffixes += ['min']
    if use_max: suffixes += ['max']
    return ["{}_{}".format(name, suffix) for suffix in suffixes]

def evaluate_model(model, trainX, trainY, textX, testY, results, 
        eval_dataset="Dev", description="Model", report_train=False):
    """
    Given 
    """
    print(description)
    model.fit(trainX, trainY)
    if report_train:
        yHat = model.predict(trainX)
        p, r, f1, s = precision_recall_fscore_support(trainY, yHat)
        results.append(["Train", description, p[1], r[1], f1[1], s[1]])
    print(devX.columns)
    yHat = model.predict(devX)
    p, r, f1, s = precision_recall_fscore_support(devY, yHat)
    results.append([eval_dataset, description, p[1], r[1], f1[1], s[1]])
    return results

