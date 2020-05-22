class GroundedSPARQLGraph: #todo
    def __init__(self, g_query):
        self.g_query = g_query

    def __str__(self):
        return self.g_query.get_query_string()
