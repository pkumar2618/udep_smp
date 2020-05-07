from gensim.models import KeyedVectors
from dbpedialib.db_utils import get_property_using_cosine_similarity


if __name__ == "__main__":

    # # get all the dbpedia_predicates
    # get_dbpedia_predicates(refresh=True)

    # covert the predicates into prefix and label
    # get_dbpedia_predicates(refresh=False)

    # vectorize the dbpedia properties using glove embedding.
    # dbpedia_property_vectorizer()

    # test cosine_smilarity the function will return top-n property based on the cosine similarity
    glove = KeyedVectors.load('./glove_gensim_mmap', mmap='r')

    # # if embedding has changed, you may need to delete the numpy_property_vector.pkl file
    vector1 = glove['Birth'].reshape((1,-1))
    vector2 = glove['place'].reshape((1, -1))
    vector = (vector1 + vector2)/2
    print(get_property_using_cosine_similarity(vector))