import csv
import re
from urllib.parse import urlsplit
from collections import OrderedDict


SEMANTICALLY_IRRELEVANT_SUFFIXES = ['\.com', '\.io', '\.co\.uk', '\.ai', '\.me', '\.json']
API_NAMES = {
        'googleapis': 'google apis',
        'adsapitwitter': 'ads api twitter',
        'graphfacebook': 'graph facebook',
        'scriptgoogleapis': 'script google apis',
        'graphvideofacebook': 'graph video facebook',
        'uploadtwitter': 'upload twitter',
        'apitwitter': 'api twitter',
        'apiadmanageryahoo': 'api ad manager yahoo',
        'mwsamazonservices': 'amazon services',
        'adsapisandboxtwitter': 'ads api sandbox twitter'
}

API_NAMES = OrderedDict(sorted(API_NAMES.items(), key=lambda t: t[0]))

def read_to_dict_array(file_name):
    results = []
    with open(file_name, 'r+') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for page in reader:
            results.append(dict(zip(headers, page)))
    return results


def find_version_end_position(path):
    pattern = r'v?\d(\.\d+)*'
    match = re.search(pattern, path)
    if match is None:
        return -1
    return match.end()

def strip_version(string):
    return re.sub(
        r'v?\d{1,3}',
        '',
        string
    )

def replace_api_names(string):
    for k,v in API_NAMES.items():
        string = string.replace(k, v)
    return string

def strip_chars(string):
    formatted_string = (
        re.sub(
            re.compile('|'.join(SEMANTICALLY_IRRELEVANT_SUFFIXES)),
            '',
            string
        )
    )
    formatted_string = (
        re.sub(
            r'[_\{\}\(\)<>\[\]\\\.\-\",\?\'\|]*',
            '',
            formatted_string
        )
    )
    return formatted_string

def strip_protocol(string):
    return re.sub(r'https:/{0,2}', '', string)

def format_request(request_example, method):
    if not re.match(r'http(s?)\:', request_example):
        request_example = 'https://' + request_example
    split_url = urlsplit(request_example)
    formatted_netloc = split_url.netloc.replace('www.', '')
    clean_path = strip_chars(split_url.path)
    version_end_pos = find_version_end_position(clean_path)
    name_portion = (clean_path[:version_end_pos])
    param_portion = (clean_path[version_end_pos:])
    name = (strip_chars(formatted_netloc)+name_portion).replace('/', ' ')
    formatted_request_example = (
        ''.join([method, ' ', name, param_portion])
    )
    no_version = strip_version(formatted_request_example.replace('/', ' '))
    proper_spaced = replace_api_names(no_version).replace('  ', ' ')
    return strip_protocol(proper_spaced)


def main():
    def format_data(example):
        example['to'] = (
            format_request(
                example['request_example'],
                example['method']
            ).lower()
        )
        example['from'] = strip_chars(example['english_example'].lower())
        return example
    with open('processed_dataset.csv', 'w+', newline='') as f:
        writer = (
            csv.DictWriter(
                f,
                delimiter=',',
                fieldnames=['db_id', 'to', 'from']
            )
        )

        formatted_examples = (
            map(format_data, read_to_dict_array('./dataset.csv'))
        )

        writer.writeheader()
        for example in formatted_examples:
            writer.writerow(
                {'db_id': example['db_id'], 'to': example['to'], 'from': example['from']}
            )


if __name__ == '__main__':
    main()
