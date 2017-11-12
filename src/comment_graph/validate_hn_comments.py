from csv import DictReader

def validate(row):
    errors = []
    try: 
        int(row['objectID'])
    except:
        errors.append("Could not convert objectID '{}' to int.".format(row['objectID']))
    try: 
        int(row['parent_id'])
    except:
        errors.append("Could not convert parent_id '{}' to int.".format(row['parent_id']))
    
    if any(errors):
        print("Errors:")
        for err in errors:
            print("  - {}".format(error))

HN_DATA = "../../data/hn_comments.csv"
with open(HN_DATA) as infile:
    reader = DictReader(infile, fieldnames=["comment_text","points","author","created_at","objectID", "parent_id"])
    for row in reader:
        validate(row)
