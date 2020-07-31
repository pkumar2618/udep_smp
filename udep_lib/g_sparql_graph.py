import logging
logger = logging.getLogger(__name__)

class GroundedSPARQLGraph: #todo
    def __init__(self, g_query_topk):
        self.g_query_topk = g_query_topk

    def __str__(self):
        return self.g_query.get_query_string()

    def run(self):
        for query in self.g_query_topk:
            query.run()
