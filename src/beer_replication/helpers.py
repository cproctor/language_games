# Helpers for generate.py

import arrow
import pandas as pd
import sqlite3
import os


def create_db_if_missing(csvfile, dbfile):
    "If the specified db does not exist, created it"
    if not os.path.exists(dbfile):
        conn = sqlite3.connect(dbfile)
        connn.text_factory = str
        for chunk in pd.read_csv(csvfile, header=None, chunksize=25000,
                names=["comment_text","points","author","created_at","object_id","parent_id"]):
            chunk.to_sql(name='comments', con=conn, if_exists='append', index=False,
            infer_datetime_format=True)
                
def split_comments_by_month(csvfile, get_monthly_filename, chunksize=10000):
    """
    Splits out a huge file full of comments into separate files, one per month.

    Params:
        csvfile: a file with all comments
        get_monthly_filename: a function mapping (year, month) -> filename
    """
    currentDf = pd.DataFrame()
    currMonthStart = arrow.get('1900').datetime
    currMonthEnd = arrow.get('1900').datetime
    counts = {}
    for chunk in pd.read_csv(csvfile, header=None, chunksize=chunksize, 
            names=["comment_text","points","author","created_at","object_id","parent_id"],
            parse_dates=["created_at"]):
        inCurrMonth = (chunk.created_at >= currMonthStart) & (chunk.created_at <= currMonthEnd)
        currChunk = chunk[inCurrMonth]
        currentDf = currentDf.append(currChunk)
        if len(currChunk) < chunksize: # We have some remainder here. Need to iterat through it
            while True:
                if len(currentDf) > 0: 
                    currentDf.to_csv(get_monthly_filename(currMonthStart.year, currMonthStart.month))
                    counts[currMonthStart] = len(currentDf)
                    print(" - {}/{} had {} comments".format(currMonthStart.year, currMonthStart.month, len(currentDf)))
                currentDf = pd.DataFrame()
                chunk = chunk[~inCurrMonth]
                if len(chunk) == 0: break # end of comments
                currMonthStart = arrow.get(chunk.iloc[0].created_at).floor('month').datetime
                currMonthEnd = arrow.get(chunk.iloc[0].created_at).ceil('month').datetime
                inCurrMonth = (chunk.created_at >= currMonthStart) & (chunk.created_at <= currMonthEnd)
                currChunk = chunk[inCurrMonth]
                currentDf = currentDf.append(currChunk)
                if len(currChunk) == 0: 
                    print(currChunk)
                    print(chunk)
                if len(currChunk) == len(chunk): break # The current month's comments may continue into the next chunk
    return counts
