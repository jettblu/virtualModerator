import requests, json
import itertools


import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_lg")


def getRelatedTopics(query):
    URL = f'http://suggestqueries.google.com/complete/search?client=firefox&q={query}'
    response = requests.get(URL)
    result = json.loads(response.content.decode('utf-8'))[1]
    return result


def getChunkData(textChunk):
    doc = nlp(textChunk)
    nounPhrases = [chunk for chunk in doc.noun_chunks]
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    return nounPhrases, verbs


def similarWordSets(wordSets, nSimilar=2):
    i = 0
    similarityDict = dict()
    for phrase1, phrase2 in wordSets:
        similarityDict[i] = phrase1.similarity(phrase2)
        i += 1
    indexes = sorted(similarityDict, key=similarityDict.get, reverse=True)[:nSimilar]
    output = []
    for index in indexes:
        for combo in wordSets[index]:
            output.append(combo)
    return output


def getRelatedQueries(text, nDesired=5):
    nounPhrases, verbs = getChunkData(textChunk=text)
    querySet = set()
    phraseCombinations = list(itertools.combinations(nounPhrases, 2))
    nounPhrasesSimilar = similarWordSets(phraseCombinations)
    phraseCombinations = list(itertools.combinations(nounPhrasesSimilar, 2))
    for combo in phraseCombinations:
        phrase = f'{combo[0]} {combo[1]}'
        relatedTopics = getRelatedTopics(phrase)
        for topic in relatedTopics:
            querySet.add(topic)
        phrase = f'{combo[1]} {combo[0]}'
        relatedTopics = getRelatedTopics(phrase)
        for topic in relatedTopics:
            querySet.add(topic)
    querySet = [nlp(topic) for topic in querySet]
    queryScores = dict()
    for phrase in nounPhrases:
        for query in querySet:
            score = phrase.similarity(query)
            queryScores[query] = queryScores.get(query, 0) + score
    queries = sorted(queryScores, key=queryScores.get)[:nDesired]
    output = []
    for query in queries:
        output.append(query)
    return output

