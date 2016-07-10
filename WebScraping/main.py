import urllib
from bs4 import BeautifulSoup
from IPython.display import Image
from IPython.display import HTML
r = urllib.urlopen('http://www.aflcio.org/Legislation-and-Politics/Legislative-Alerts').read()
soup = BeautifulSoup(r,"lxml")
print type(soup)
print soup.prettify()[0:1000]

letters = soup.find_all("div",class_="info" )
print(letters)
Image('http://www.openbookproject.net/tutorials/getdown/css/images/lesson4/HTMLDOMTree.png')
Image('http://www.cs.toronto.edu/~shiva/cscb07/img/dom/treeStructure.png')
HTML('<iframe src=http://www.aflcio.org/Legislation-and-Politics/Legislative-Alerts width=700 height=500></iframe>')
print soup.prettify()[28700:30500]
letters = soup.find_all("div", class_="ec_statements")
print type(letters)
letters[0]
lobbying = {}
for element in letters:
    lobbying[element.a.get_text()] = {}
letters[0].a["href"]
prefix = "www.aflcio.org"
for element in letters:
    lobbying[element.a.get_text()]["link"] = prefix + element.a["href"]
letters[0].find(id="legalert_date")
for element in letters:
    date = element.find(id="legalert_date").get_text()
    lobbying[element.a.get_text()]["date"] = date
for item in lobbying.keys():
    print item + ": " + "\n\t" + "link: " + lobbying[item]["link"] + "\n\t" + "date: " + lobbying[item]["date"] + "\n\n"

import os, csv
os.chdir("C:\Users\Freeware Sys\Desktop\Project\WebScraping")

with open("lobbying.csv", "w") as toWrite:
    writer = csv.writer(toWrite, delimiter=",")
    writer.writerow(["name", "link", "date"])
    for a in lobbying.keys():
        writer.writerow([a.encode("utf-8"), lobbying[a]["link"], lobbying[a]["date"]])

    import json

    with open("lobbying.json", "w") as writeJSON:
        json.dump(lobbying, writeJSON)
