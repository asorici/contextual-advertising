from graphlab.cython.context import debug_trace

__author__ = 'andrei'

import numpy as np
from lxml import etree
import urllib2
import xml.etree.ElementTree as ET
from lxml.html.soupparser import fromstring

class WebsiteDataExtractor:
    parser = etree.HTMLParser()
    defaultPaths = {}
    customPaths = {}
    CONCAT_ANSWERS = ["title", "summary"]
    GROUP_BY_CHILDREN = ["mainText"]

    def __init__(self, definitionsFile):
        tree = ET.parse(definitionsFile)
        root = tree.getroot()
        for child in root:
            if child.tag == "default":
                for elemPath in child:
                    self.defaultPaths[elemPath.tag] = elemPath.text
            else:
                domain = ""
                parsers = {}
                for elemPath in child:
                    if elemPath.tag == "domain":
                        domain = elemPath.text
                    else:
                        parsers[elemPath.tag] = elemPath.text

                if domain:
                    self.customPaths[domain] = parsers


    def getElementPaths(self, url):
        for domain in self.customPaths.keys():
            if domain in url:
                return self.customPaths[domain]

        return self.defaultPaths

    @staticmethod
    def clean_string(text):
        import re

        #keep only numbers and letters
        text = u''.join(map(lambda x: x if ( (str.isalnum(x) if isinstance(x, str) else unicode.isalnum(x)) or x == " ") else " ", text))

        #lowercase only
        text = text.lower()

        multiSpacePattern = re.compile(r'\s+')
        text = re.sub(multiSpacePattern, " ", text)

        text = re.sub(r"<script>.*<\/script>", " ", text)

        return text.strip()

    @staticmethod
    def tokenizeWebsiteUrl(url):
        return ""

    def crawlPage(self, page):
        tree = etree.fromstring(urllib2.urlopen(page).read(), self.parser)
        paths = self.getElementPaths(page)

        pageData = {"urlTokens" : self.tokenizeWebsiteUrl(page)}
        for el, elPattern in paths.iteritems():
            s = tree.xpath(elPattern)

            if el in self.CONCAT_ANSWERS:
                s = self.clean_string(" ".join(s))
            elif el in self.GROUP_BY_CHILDREN:
                ls = []
                for aux in s:
                    if (isinstance(aux, etree._Element)):
                        val = aux.xpath('.//text()')
                        for m in val:
                            if (m.isspace() or not m):
                                val.remove(m)
                        if (val):
                            ls = ls + [val]
                s = map(lambda x: " ".join(x), ls)
                s = map(self.clean_string, s)
            else:
                s = map(self.clean_string, s)
                s = filter(None, s)

            pageData[el] = s

        return pageData





test = WebsiteDataExtractor("/Users/andrei/Documents/SIEN/contextual-advertisingPyP/contextual-advertising/dataset/WebsiteElementsPathDef.xml")
d = test.crawlPage("http://www.generation-nt.com/rechauffement-climatique-ere-glaciaire-retard-actualite-1923734.html")

