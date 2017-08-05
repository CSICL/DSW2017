import requests
from bs4 import BeautifulSoup
import urllib3

http = urllib3.PoolManager()
url = "http://www.example.com"
response =http.request('GET',url)
soup = BeautifulSoup(response.data)


print soup


# Function syntax
def getWebPage() :
	http = urllib3.PoolManager()
	url = "http://www.example.com"
	response =http.request('GET',url)
	soup = BeautifulSoup(response.data)

	print soup


if __name__ == '__main__':
	getWebPage()


# Passing url as function argument
def getWebPage(url) :
	http = urllib3.PoolManager()
	response =http.request('GET',url)
	soup = BeautifulSoup(response.data)

	print soup


if __name__ == '__main__':
	url = "http://www.example.com"
	getWebPage(url)

# Passing list of url as function argument
def getWebPage(url) :
	http = urllib3.PoolManager()
	response =http.request('GET',url)
	soup = BeautifulSoup(response.data)

	print soup


if __name__ == '__main__':
	urls = ["http://www.example.com","https://www.wikipedia.org/"]
	for url in urls:
		getWebPage(url)

#Returning content from function
def getWebPage(url) :
	http = urllib3.PoolManager()
	response =http.request('GET',url)
	soup = BeautifulSoup(response.data)

	return soup


if __name__ == '__main__':
	url = "http://www.example.com"
	content = getWebPage(url)
	print content


