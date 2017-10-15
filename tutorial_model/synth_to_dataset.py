import agate
import random
import os
from sys import exit, argv

if len(argv) < 2:
    print('Need input path for dataset.')
    exit(-1)

OUT_PATH = './data_files'

column_names = ['input', 'output']
column_types = [agate.Text(), agate.Text()]

table = agate.Table.from_csv(argv[1]+'/synthdata.csv', column_names, column_types)
table.order_by(lambda row: random.random())

count = len(table)

train_amount = int(count*0.7)
validate_amount = int(count*0.2)
test_amount = int(count*0.1)

test_table = table.limit(0, stop=test_amount)
validate_table = table.limit(test_amount, stop=validate_amount)
train_table = table.limit(validate_amount, stop=train_amount)

def write_to_files(tables, path):
    for table in tables:
        name, data = table
        in_table = data.exclude('output')
        out_table = data.exclude('input')
        in_table.to_csv(path+'/'+name+'_in.csv')
        out_table.to_csv(path+'/'+name+'_out.csv')


if not os.path.isdir(OUT_PATH):
    try:
        os.mkdir(OUT_PATH)
    except OSError as e:
        print(e)
        exit(-1)

file_names = ['training', 'validation', 'test']

tables = [train_table, validate_table, test_table]

write_to_files(zip(file_names, tables), OUT_PATH)






