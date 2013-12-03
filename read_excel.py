import xlrd

def string_from_excel_column(col):
	if col == 0:
		return 'id'
	elif col == 1:
		return 'username'
	elif col == 2:
		return 'universal_time_stamp'
	elif col == 3:
		return 'local_time_stamp'
	elif col == 4:
		return 'text'
	elif col == 5:
		return 'relevance'
	elif col == 6:
		return 'content_source'
	elif col == 7:
		return 'language'
	else:
		return None


diabetes_excel1 = "DiabetesMay7.2.13.xlsx"
diabetes_excel2 = "Diabetes40413to41713.xlsx"
diabetes_excel3 = "Diabetes4.4.13to4.24.13.xlsx"

book1 = xlrd.open_workbook(diabetes_excel1)
book2 = xlrd.open_workbook(diabetes_excel2)
book3 = xlrd.open_workbook(diabetes_excel3)

rows1 = book1.sheet_by_index(0).nrows
rows2 = book2.sheet_by_index(0).nrows
rows3 = book3.sheet_by_index(0).nrows

number_of_rows = rows1 + rows2 + rows3

print "The total number of tweets / rows is: ", number_of_rows
# print "The number of worksheets (1) is", book1.nsheets
# print "Worksheet (1) name(s):", book1.sheet_names()

data_sheets = [book1.sheet_by_index(0), book2.sheet_by_index(0), \
    book3.sheet_by_index(0)]

# print "Cell D30 (!) is", sh.cell_value(rowx=29, colx=3)

# parse through the data in each sheet!
tweets_per_sheet = {}
for shx in range(len(data_sheets)):
    sh = data_sheets[shx]
    tweets = [] # for each sheet, make a new tweets array
    # for rx in range(sh.nrows): UNCOMMENT FOR REAL DEAL
    for rx in range(10):
    	if rx == 0:
    		continue
        tweet = {} #for each row, create a tweet object / dictionary
        for cx in range(sh.ncols):
            cell = sh.cell_value(rowx=rx, colx=cx)
            key = string_from_excel_column(cx)
            if key is None:
        	    break
            else:
                tweet[key] = cell
        # print 'tweet ', rx, ': ', tweet['text']
        tweets.append(tweet)

    print 'size of tweets array: ', len(tweets)
    if shx == 0:
        tweets_per_sheet[diabetes_excel1] = tweets
    elif shx == 1:
        tweets_per_sheet[diabetes_excel2] = tweets
    else:
        tweets_per_sheet[diabetes_excel3] = tweets

for key in tweets_per_sheet.keys():
	tweets_array = tweets_per_sheet[key]
	print 'number of tweets for sheet', key, ': ', len(tweets_array)
