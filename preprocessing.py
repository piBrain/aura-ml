import csv
import re
from urllib.parse import urlsplit


SEMANTICALLY_IRRELEVANT_SUFFIXES = ['\.com', '\.io', '\.co\.uk', '\.ai', '\.me']


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


def format_request(request_example, method):
    if not re.match(r'http(s?)\:', request_example):
        request_example = 'https://' + request_example
    split_url = urlsplit(request_example)
    formatted_netloc = split_url.netloc.replace('www.', '')
    path = split_url.path
    version_end_pos = find_version_end_position(path)
    name_portion = ''.join(path[:version_end_pos])
    param_portion = ''.join(path[version_end_pos:])
    formatted_netloc = (
        re.sub(
            re.compile('|'.join(SEMANTICALLY_IRRELEVANT_SUFFIXES)),
            '',
            formatted_netloc
        )
    ).replace('.', '').replace('-', '')
    name = (formatted_netloc+name_portion).replace('/', '')
    formatted_request_example = (
        ''.join([method, ' ', name, param_portion])
    )
    return formatted_request_example.replace('/', ' ')


def main():
    def format_data(example):
        example['from'] = (
            format_request(
                example['request_example'],
                example['method']
            )
        )
        example['to'] = example['english_example'].lower()
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
