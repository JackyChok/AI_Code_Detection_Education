from __future__ import print_function
import sys
import atexit
from os import path
from json import dumps, loads
import csv
import requests


# function to fetch data from API based on a given parameter
def api_request(text):
    # Set the request headers

    api_key = 'API key here'
    base_url = 'https://api.sapling.ai/api/v1/aidetect'
    headers = {
        "Content-Type": "application/json",
    }

    # Set the request data
    data = {
        "key": api_key,
        "text": text
    }

    # Make the POST request
    response = requests.post(base_url, headers=headers, json=data)

    # Get the response message
    response_message = response.text

    # Check if the request was successful
    if response.status_code == 200:
        print("POST request was successful.")
        print("Response message:", response_message)
    else:
        print(f"Error: {response.status_code} - {response_message}")

    return (response.json())['score']


# filename
filename = "./rerun_variants/variant_11_fullv2.csv"
filename2 = "progress.txt"
  
# 1. whenever something break, remove that row from good_answer.csv
# 2. increment the file number 1

base_url = "https://sapling.ai/ai-content-detector"
x_path_input = "//*[@id='content-editor']"
x_path_output = "//*[@id='fake-prob']"


def read_counter():
    return loads(open("counter.json", "r").read()) + 1 if path.exists("counter.json") else 0


def write_counter():
    with open("counter.json", "w") as f:
        f.write(dumps(counter))


counter = read_counter()
atexit.register(write_counter)
filename1 = f"./detector_given_ai{counter}.txt"
counter_reset_god = 0
print(filename1)


def selenium_saping(row_no_human, row_no_ai):
    with open(filename, mode='r', encoding="utf8") as csv_file, \
            open(filename1, mode='w+') as out_file:
        # open(filename2, mode='r+') as progress_file, \
        reader = csv.reader(csv_file)
        # Skip header
        next(reader, None)
        # write header
        out_file.writelines(",binary_human,human,binary_ai,ai\n")
        # progreses = progress_file.readlines()

        for row in reader:
            try:
                # get the first item of the row (assuming it's in the first column)
                # HUMAN---------------------
                item = row[row_no_human]

                binary_human = "placeholder"

                # AI------------------------
                item = row[row_no_ai]
     
                try:
                    output_value2 = api_request(item)
                except:
                    output_value2 = False
                
                if not output_value2:
                    output_value2 = '0.0'
                    print(row[0], '0.0', "ai code less than 50 char")

                binary_ai = 1
                if float(output_value2) > 0.90:
                    binary_ai = 0

                out_file.writelines(f'{row[0]}, {binary_human},output_value1, {binary_ai},{output_value2}\n')
                print(row[0], "--->", binary_human, 'place_holder', "|", binary_ai, output_value2)

            except Exception as e:
                print(e)
                sys.exit("Shit happens")
                # continue


# EXECUTE: Input Human code and AI code into detector
print("HUMAN IN AND AI IN")

selenium_saping(4, 5)

print("--Complete--")


