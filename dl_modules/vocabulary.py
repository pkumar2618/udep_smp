class Vocabulary(object):
    """
    this Class maintains token to index mapping and index to token,
    which can also be done with gensim is suppose.
    """
    # pass
    def __init__(self, token_to_index=None, add_unk=True, unk_token="<UNK>"):
        """
        :param token_to_index: a dictionary of token-index pairs, as a new token adds, index
        is increased.
        :param add_unk: Initialize to handle unknown tokens.
        :param unk_token: the unknown token is <UNK>
        """
        if token_to_index == None:
            self._token_to_index = {}

        self._token_to_index = token_to_index
        self._index_to_token = {index: token
                                for token, index in self._token_to_idx.items()}
        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1 # in case the unk_token not added.
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def to_serializable(self):
        """
        :return: Return a dictionary than can be serialized.
        """
        return {'token_to_index': self._token_to_index, 'add_unk': self._add_unk,
                'unk_token': self._unk_token}

    @classmethod
    def from_serializable(cls, vocab):
        """
        create the vocabulary from
        :return:
        """
        return cls(**vocab)

    def add_token(self, token):
        """
        Update Dictionary to include the new token.
        :param token:
        :return: Return index of the token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def lookup_token(self,token):
        """
        look up the token and return it's index, if the token is not found then return
        index of token <UNK>
        :param token:
        :return: index of the token if found in dictionary, otherwise return index of unk_token
        """
        if self._add_unk:
            return self._token_to_index.get(token, self.unk_index)
        else:
            return self._token_to_index[token]

    def lookup_index(self, index):
        """
        given index return it's corresponding token
        :param index:
        :return: return the corresponding token
        """
        if index not in self._index_to_token:
            raise KeyError("the index (%d) not in the Vocabulary" % index)

        return self._index_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)